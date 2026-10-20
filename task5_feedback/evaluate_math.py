from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"] #sft = the untouched base model, the other two are the supplied frozen adapters
#the pairwise comparisons scored by the ai judge -> each trained policy against sft (the manual's win rate against
#sft), and the two trained policies against each other
COMPARISONS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlaif", "rlvr")]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def generate_responses(cfg, rows, tokenizer, name):
    #one greedy response per problem (do_sample=False), so the three policies answer the same fixed problems in the
    #same order with no sampling noise between them. the problem text is the row's own messages, which already ask
    #for the designated '#### <number>' final line
    policy = load_frozen_policy(cfg, name)
    size, records = int(cfg["math_batch_size"]), []
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        out = batch_generate(
            policy, tokenizer, [prompt_messages(row) for row in chunk],
            max_prompt_length=int(cfg["math_max_prompt_length"]),
            max_new_tokens=int(cfg["math_max_new_tokens"]),
            do_sample=False,
        )
        for i, row in enumerate(chunk):
            response = out["responses"][i]
            predicted = extract_designated_final(response) #None when there is no '#### <number>' field at all
            records.append({
                "problem_id": str(row["source_index"]),
                "policy": name,
                "question": row["question"],
                "gold_final": str(row["gold_final"]),
                "problem_type": row.get("svamp_type"), #svamp only -> addition, subtraction, ...
                "response": response,
                "response_tokens": out["response_lengths"][i],
                "truncated": out["truncated"][i], #hit the 512 token cap without finishing
                "predicted_final": predicted,
                "format_ok": predicted is not None, #format compliance -> the designated final answer could be parsed
                "exact": exact_reward(response, str(row["gold_final"])), #the rlvr reward, 1.0 only if it matches the gold answer
            })
        print(f"  [{name}] generated {min(start + size, len(rows))}/{len(rows)}", flush=True)
    clear_gpu(policy)
    return records


def failure_type(record):
    #one mutually exclusive outcome per response, for the failure-type comparison
    if record["exact"] == 1.0:
        return "correct"
    if record["format_ok"]:
        return "wrong_final_answer"
    return "truncated_no_final" if record["truncated"] else "missing_final_format"


def judge_comparisons(judge, responses):
    #for every problem and every policy pair the fixed judge says which response it prefers. the judge class already
    #balances the a/b order deterministically and caches every verdict on disk
    pairs = []
    for first, second in COMPARISONS:
        for a, b in zip(responses[first], responses[second]):
            if a["response"] == b["response"]:
                preference = "TIE" #identical text cannot be ranked, counted as a tie without spending a judge call
            else:
                preference = judge.compare(a["question"], a["response"], b["response"])
            #the exact verifier's view of the same pair -> it prefers whichever response is correct, and ties otherwise
            verifier = "A" if a["exact"] > b["exact"] else "B" if b["exact"] > a["exact"] else "TIE"
            pairs.append({
                "problem_id": a["problem_id"], "first": first, "second": second,
                "judge_preference": preference, #A = the first policy's response, B = the second's
                #clean / unparsed / multiple_labels, see PairwiseAIJudge.output_kind. identical texts were never judged
                "judge_output": "not_judged" if a["response"] == b["response"] else judge.output_kind(a["question"], a["response"], b["response"]),
                "verifier_preference": verifier,
                "identical_responses": a["response"] == b["response"],
                "first_exact": a["exact"], "second_exact": b["exact"],
                #win = 1, tie = 0.5, loss = 0 for the first policy, the manual's pairwise scoring
                "first_score": {"A": 1.0, "TIE": 0.5, "B": 0.0}[preference],
            })
        print(f"  judged {first} vs {second}", flush=True)
    return pairs


def summarise(dataset, responses, pairs):
    pairs = pd.DataFrame(pairs)
    rows = []
    for name in POLICIES:
        frame = pd.DataFrame(responses[name])
        row = {
            "dataset": dataset,
            "policy": name,
            "n": len(frame),
            "exact_accuracy": frame["exact"].mean(),
            "format_compliance": frame["format_ok"].mean(),
            "length_mean": frame["response_tokens"].mean(),
            "length_std": frame["response_tokens"].std(ddof=0),
            "truncation_rate": frame["truncated"].mean(),
        }
        row.update({f"failure_{k}": v for k, v in frame.apply(failure_type, axis=1).value_counts(normalize=True).items()})
        #the ai pairwise result against sft. sft against itself is 0.5 by definition and is left empty
        versus = pairs[(pairs["first"] == name) & (pairs["second"] == "sft")]
        if len(versus):
            row.update({
                "win_rate_vs_sft": versus["first_score"].mean(),
                "wins_vs_sft": int((versus["judge_preference"] == "A").sum()),
                "ties_vs_sft": int((versus["judge_preference"] == "TIE").sum()),
                "losses_vs_sft": int((versus["judge_preference"] == "B").sum()),
                "identical_to_sft": int(versus["identical_responses"].sum()),
            })
        rows.append(row)

    agreement = []
    for first, second in COMPARISONS:
        frame = pairs[(pairs["first"] == first) & (pairs["second"] == second)]
        #pairs where the verifier can tell the two apart -> exactly one of the two responses is correct
        decided = frame[frame["verifier_preference"] != "TIE"]
        agreement.append({
            "dataset": dataset, "comparison": f"{first}_vs_{second}", "n": len(frame),
            "win_rate_first": frame["first_score"].mean(),
            "judge_ties": int((frame["judge_preference"] == "TIE").sum()),
            #ambiguous judge outputs, reported apart from the ties -> no label at all (counted as TIE by the parser), or
            #several labels at once (the parser kept the first, so these sit among the wins and losses)
            "judge_unparsed": int((frame["judge_output"] == "unparsed").sum()),
            "judge_multiple_labels": int((frame["judge_output"] == "multiple_labels").sum()),
            "identical_not_judged": int((frame["judge_output"] == "not_judged").sum()),
            #verifier-judge agreement -> both give the same one of the three outcomes (first / second / tie)
            "agreement": (frame["judge_preference"] == frame["verifier_preference"]).mean(),
            "verifier_decided_pairs": len(decided),
            #on the pairs the verifier decides, how the judge treats the correct response
            "judge_prefers_correct": (decided["judge_preference"] == decided["verifier_preference"]).mean(),
            "judge_ties_on_decided": (decided["judge_preference"] == "TIE").mean(),
            "judge_prefers_wrong": ((decided["judge_preference"] != decided["verifier_preference"])
                                    & (decided["judge_preference"] != "TIE")).mean(),
            #on the pairs the verifier cannot separate (both right or both wrong), how often the judge still picks one
            "judge_decides_on_verifier_ties": (frame[frame["verifier_preference"] == "TIE"]["judge_preference"] != "TIE").mean(),
        })
    return pd.DataFrame(rows), pd.DataFrame(agreement)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    #1 -> matched responses from the three frozen policies. nothing is trained in this task, both adapters are supplied
    responses = {}
    for name in POLICIES:
        path = outdir / f"generated_{args.dataset}_{name}.jsonl"
        if path.exists():
            print(f"[{name}] responses already generated, skipping")
            responses[name] = read_jsonl(path)
        else:
            responses[name] = generate_responses(cfg, rows, tokenizer, name)
            write_jsonl(path, responses[name])

    #2 -> the fixed pairwise ai judge, loaded only after the policies are released from the gpu
    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
    pairs = judge_comparisons(judge, responses)
    write_jsonl(outdir / f"pairwise_{args.dataset}.jsonl", pairs)

    #3 -> exact accuracy, format compliance, length, failure types, pairwise result and verifier-judge agreement
    summary, agreement = summarise(args.dataset, responses, pairs)
    summary.to_csv(outdir / f"summary_{args.dataset}.csv", index=False)
    agreement.to_csv(outdir / f"agreement_{args.dataset}.csv", index=False)
    print(summary.set_index("policy").T.round(4).to_string())
    print(agreement.set_index("comparison").T.round(4).to_string())


if __name__ == "__main__":
    main()
