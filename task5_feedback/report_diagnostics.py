from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task5_feedback.evaluate_math import COMPARISONS, POLICIES
from task5_feedback.rlvr import extract_designated_final, numerically_equal
from task5_feedback.score_perturbations import EXPECTED_VARIANTS, PERTURBED, load_diagnostic_groups

#report diagnostics for task 5 -> pure table work on files that already exist (generated responses, pairwise and
#diagnostic records, the judge's cached raw outputs). no policy and no judge is loaded, nothing existing is rewritten.
#run from the repo root ->  python -m task5_feedback.report_diagnostics --config configs/feedback.yaml

DATASETS = ["gsm", "transfer"]
POLICY_COLORS = {"sft": "#52514e", "rlvr": "#2a78d6", "rlaif": "#eb6834"} #same palette as task5.ipynb
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
#the non-parseable categories, checked in this order so every response lands in exactly one
FORMAT_CATEGORIES = ["truncated", "hash_dollar", "hash_text", "boxed", "bold_number", "other_prose"]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})


def last_number(text):
    #the last number written anywhere in the response, commas dropped ("4,000" -> "4000"), None if there is none
    found = NUMBER_RE.findall(str(text))
    return found[-1].replace(",", "") if found else None


def lenient_correct(record):
    #supplementary diagnostic, NOT the official metric -> a parseable response keeps the verifier's verdict, a
    #non-parseable one counts as correct when the last number it writes equals the gold answer. caveat: that can
    #credit a gold number that only appears as an intermediate value, which the strict verifier is designed to refuse
    if record["format_ok"]:
        return float(record["exact"])
    guess = last_number(record["response"])
    return float(guess is not None and numerically_equal(guess, record["gold_final"]))


def format_category(record):
    #why a response failed the designated '#### <number>' parse -> the first matching pattern wins
    text = record["response"]
    if record["truncated"]:
        return "truncated" #ran into the 512 token cap before writing any final
    if re.search(r"####\s*\$\s*\d", text):
        return "hash_dollar" #'#### $300', the regex wants the digits right after ####
    if "####" in text:
        return "hash_text" #'#### Answer: 50', '#### Final Answer: ...'
    if "\\boxed" in text:
        return "boxed" #latex \boxed{144}
    if re.search(r"\*\*[^*]*\d[^*]*\*\*", text):
        return "bold_number" #markdown **9** somewhere in the conclusion
    return "other_prose"


def mcnemar_exact(b, c):
    #exact two-sided mcnemar test on the discordant counts -> under no difference each discordant problem is a fair
    #coin, so p = P(binomial(b + c, 1/2) at least as far from the middle as observed)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def load_responses(outdir):
    return {(ds, name): read_jsonl(outdir / f"generated_{ds}_{name}.jsonl") for ds in DATASETS for name in POLICIES}


def accuracy_breakdown(responses):
    #1 -> strict accuracy (official), format compliance, accuracy among the parseable responses, the lenient
    #supplementary accuracy, and what the non-parseable responses actually contain
    rows = []
    for (ds, name), records in responses.items():
        parseable = [r for r in records if r["format_ok"]]
        failed = [r for r in records if not r["format_ok"]]
        categories = pd.Series([format_category(r) for r in failed]).value_counts()
        row = {
            "dataset": ds, "policy": name, "n": len(records),
            "strict_accuracy": np.mean([r["exact"] for r in records]),
            "format_compliance": len(parseable) / len(records),
            "non_parseable": len(failed),
            "parseable_accuracy": np.mean([r["exact"] for r in parseable]),
            "lenient_accuracy": np.mean([lenient_correct(r) for r in records]),
            #non-parseable responses whose last number is the gold answer -> the gap between strict and lenient
            "non_parseable_last_number_gold": int(sum(lenient_correct(r) for r in failed)),
        }
        row.update({f"format_{c}": int(categories.get(c, 0)) for c in FORMAT_CATEGORIES})
        rows.append(row)
    return pd.DataFrame(rows)


def policy_vs_sft(responses):
    #2 -> the trained policies against sft on the same problems. most greedy answers are word-for-word sft's, so the
    #comparison lives in the discordant problems (one right, the other wrong), tested with the exact mcnemar test
    rows = []
    for ds in DATASETS:
        base = responses[(ds, "sft")]
        for name in ["rlvr", "rlaif"]:
            other = responses[(ds, name)]
            row = {"dataset": ds, "policy": name, "n": len(base),
                   "identical_to_sft": sum(a["response"] == b["response"] for a, b in zip(other, base))}
            for metric, score in [("strict", lambda r: r["exact"]), ("lenient", lenient_correct)]:
                gained = sum(score(a) > score(b) for a, b in zip(other, base)) #right now, wrong under sft
                lost = sum(score(a) < score(b) for a, b in zip(other, base)) #wrong now, right under sft
                row.update({f"{metric}_gained": int(gained), f"{metric}_lost": int(lost),
                            f"{metric}_mcnemar_p": mcnemar_exact(int(gained), int(lost))})
            rows.append(row)
    return pd.DataFrame(rows)


def judge_key(cfg, problem, a, b):
    #the cache key PairwiseAIJudge._key builds, recomputed so every cached verdict can be traced to where it was asked
    payload = json.dumps({"model": cfg["ai_judge_model"], "problem": problem, "a": a, "b": b}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def raw_kind(raw):
    #what the judge literally wrote, before the release parser turned it into a verdict
    labels = set(re.findall(r"\b(A|B|TIE)\b", raw))
    if not labels:
        return "unparsed"
    if len(labels) > 1:
        return "multi_label"
    return labels.pop() #'A', 'B' or 'TIE'


def judge_sources(cfg, responses, groups):
    #every judge call the task 5 scripts made, keyed exactly like the cache -> {key: source}. a key asked twice (e.g.
    #rlaif and rlvr wrote the same text, so rlaif-vs-sft repeats an rlvr-vs-sft pair) was served from the cache the
    #second time, so each distinct call is attributed to the first comparison that asked it (setdefault)
    sources = {}
    for ds in DATASETS:
        for first, second in COMPARISONS:
            for a, b in zip(responses[(ds, first)], responses[(ds, second)]):
                if a["response"] != b["response"]: #identical texts were tied without a call
                    sources.setdefault(judge_key(cfg, a["question"], a["response"], b["response"]), f"{ds}_{first}_vs_{second}")
    for variants in groups.values():
        clean = variants["clean_correct"]
        for v in PERTURBED: #the controlled clean-vs-perturbed pairs, clean is always in slot A
            sources.setdefault(judge_key(cfg, clean["question"], clean["response"], variants[v]["response"]), "diagnostic_pairs")
        order = sorted(EXPECTED_VARIANTS) #the group-reward round robin, same order as score_perturbations.group_rewards
        for i, j in itertools.combinations(range(len(order)), 2):
            key = judge_key(cfg, clean["question"], variants[order[i]]["response"], variants[order[j]]["response"])
            sources.setdefault(key, "diagnostic_group_only") #the clean-vs-x calls were already counted as pairs
    return sources


def judge_output_breakdown(cfg, outdir, responses, groups):
    #3 -> the raw outputs, by where they were asked. a verdict's position is the RAW slot the judge named, before the
    #release's hash swap maps it back, so this is where a position preference shows
    raw = json.loads((outdir / "judge_cache_raw.json").read_text(encoding="utf-8"))
    sources = judge_sources(cfg, responses, groups)
    frame = pd.DataFrame([{"key": k, "raw": v, "kind": raw_kind(v), "source": sources.get(k, "unmatched")} for k, v in raw.items()])
    rows = []
    for source, sub in [("all", frame)] + list(frame.groupby("source")):
        counts = sub["kind"].value_counts()
        non_tie = int(len(sub) - counts.get("TIE", 0))
        rows.append({"source": source, "calls": len(sub),
                     "tie": int(counts.get("TIE", 0)), "raw_a": int(counts.get("A", 0)), "raw_b": int(counts.get("B", 0)),
                     "multi_label": int(counts.get("multi_label", 0)), "unparsed": int(counts.get("unparsed", 0)),
                     "tie_rate": counts.get("TIE", 0) / len(sub),
                     #among the verdicts that were not a tie, the share the parser had to resolve from several labels
                     "multi_label_share_of_non_tie": counts.get("multi_label", 0) / non_tie if non_tie else float("nan")})
    return pd.DataFrame(rows), frame


def visible_text(response):
    #the response with the <<expression=result>> calculator annotations removed -> what a reader actually sees
    return re.sub(r"<<[^>]*>>", "", response)


def diagnostic_detail(outdir, groups, judge_frame, cfg):
    #4 -> the controlled pairs per perturbation and mechanism, with counts, plus how many decisive judge verdicts
    #came from a multi-label output the parser resolved, and whether each reasoning corruption is visible at all
    pairs = pd.DataFrame(read_jsonl(outdir / "diagnostic_pairs.jsonl"))
    kind = dict(zip(judge_frame["key"], judge_frame["kind"]))
    pairs["raw_kind"] = [kind.get(judge_key(cfg, r["question"], r["clean_response"], r["variant_response"])) for _, r in pairs.iterrows()]
    hidden = {pid: visible_text(v["corrupt_reasoning_correct_final"]["response"]) == visible_text(v["clean_correct"]["response"])
              for pid, v in groups.items()}
    rows = []
    for variant in PERTURBED:
        sub = pairs[pairs["variant_type"] == variant]
        for mechanism in ["verifier", "judge"]:
            counts = sub[mechanism].value_counts()
            decisive = sub[sub[mechanism] != "tie"]
            rows.append({"variant_type": variant,
                         "mechanism": "rlvr_exact_verifier" if mechanism == "verifier" else "rlaif_pairwise_judge",
                         "pairs": len(sub), "better": int(counts.get("better", 0)), "tie": int(counts.get("tie", 0)),
                         "wrong": int(counts.get("wrong", 0)),
                         "decisive_from_multi_label": int((decisive["raw_kind"] == "multi_label").sum()) if mechanism == "judge" else 0,
                         #reasoning-corruption only -> the corruption sits entirely inside hidden calculator tags
                         "corruption_only_in_calculator_tags": int(sum(hidden.values())) if variant == "corrupt_reasoning_correct_final" else None})
    detail = pd.DataFrame(rows)
    #what the rlaif group reward looks like when the five variants are scored together -> all 0.5 means every one of
    #the ten pairwise comparisons was a tie, ie a zero-std group that gives no gradient
    group = pd.read_csv(outdir / "diagnostic_group_rewards.csv")
    per_problem = group.groupby("problem_id")["rlaif_reward"]
    group_stats = {"groups": int(per_problem.ngroups),
                   "groups_all_rlaif_0_5": int((per_problem.apply(lambda r: np.allclose(r, 0.5))).sum()),
                   "groups_zero_std": int((per_problem.std(ddof=0) == 0).sum()),
                   "visible_corruption_problems": sorted(str(p) for p, h in hidden.items() if not h)}
    return detail, group_stats


def instruction_position(messages):
    #where the '#### <number>' instruction sits in the user message -> before or after the problem text
    text = messages[-1]["content"]
    at = text.find("####")
    return "after_problem" if at > len(text) / 2 else "before_problem"


def transfer_shift(cfg, responses, accuracy):
    #5 -> what differs between the in-domain and the transfer set besides the domain itself
    out = {}
    for ds, key in [("gsm", "gsm_eval"), ("transfer", "math_transfer_eval")]:
        rows = read_jsonl(cfg["paths"][key])
        acc = accuracy[accuracy["dataset"] == ds].set_index("policy")
        out[ds] = {
            "problems": len(rows),
            "question_words_mean": float(np.mean([len(r["question"].split()) for r in rows])),
            "instruction_position": pd.Series([instruction_position(r["messages"]) for r in rows]).value_counts().to_dict(),
            "example_prompt": rows[0]["messages"][-1]["content"],
            "format_compliance": acc["format_compliance"].round(4).to_dict(),
            "parseable_accuracy": acc["parseable_accuracy"].round(4).to_dict(),
            "strict_accuracy": acc["strict_accuracy"].round(4).to_dict(),
            "response_tokens_mean": {name: float(np.mean([r["response_tokens"] for r in responses[(ds, name)]])) for name in POLICIES},
        }
    return out


def plot_accuracy(accuracy, path):
    #strict (official) next to the two supplementary readings, per policy, gsm8k and svamp side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    columns = [("strict_accuracy", "strict (official)"), ("parseable_accuracy", "among parseable"), ("lenient_accuracy", "lenient (last number)")]
    positions = np.arange(len(columns))
    for axis, ds, title in zip(axes, DATASETS, ["gsm8k (in-domain)", "svamp (transfer)"]):
        frame = accuracy[accuracy["dataset"] == ds].set_index("policy")
        for offset, name in zip([-0.26, 0.0, 0.26], POLICIES):
            bars = axis.bar(positions + offset, [frame.loc[name, c] for c, _ in columns], width=0.24, color=POLICY_COLORS[name], label=name)
            axis.bar_label(bars, fmt="%.2f", padding=3, color="#0b0b0b", fontsize=8)
        axis.set_xticks(positions, [label for _, label in columns])
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of problems")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_judge(breakdown, path):
    #raw judge outputs per source as stacked shares -> tie / raw slot A / raw slot B / several labels
    order = ["gsm_rlvr_vs_sft", "gsm_rlaif_vs_sft", "gsm_rlaif_vs_rlvr", "transfer_rlvr_vs_sft", "transfer_rlaif_vs_sft",
             "transfer_rlaif_vs_rlvr", "diagnostic_pairs", "diagnostic_group_only", "all"]
    frame = breakdown.set_index("source").reindex([s for s in order if s in set(breakdown["source"])])
    kinds = [("tie", "TIE", "#c3c2b7"), ("raw_a", "A (first slot)", "#2a78d6"), ("raw_b", "B (second slot)", "#eb6834"),
             ("multi_label", "several labels", "#7a5af5")]
    fig, axis = plt.subplots(figsize=(10, 4.2))
    left = np.zeros(len(frame))
    labels = [f"{s.replace('_', ' ')} (n={int(n)})" for s, n in zip(frame.index, frame["calls"])]
    for column, label, color in kinds:
        share = (frame[column] / frame["calls"]).to_numpy()
        axis.barh(labels, share, left=left, color=color, edgecolor="#fcfcfb", linewidth=1.5, label=label)
        left += share
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("share of judge calls (raw output, before the a/b swap is undone)")
    axis.grid(axis="y", visible=False)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    responses = load_responses(outdir)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])

    accuracy = accuracy_breakdown(responses)
    accuracy.to_csv(outdir / "accuracy_breakdown.csv", index=False)
    print(accuracy.round(4).to_string(index=False))

    versus = policy_vs_sft(responses)
    versus.to_csv(outdir / "policy_vs_sft.csv", index=False)
    print(versus.round(4).to_string(index=False))

    breakdown, judge_frame = judge_output_breakdown(cfg, outdir, responses, groups)
    breakdown.to_csv(outdir / "judge_output_breakdown.csv", index=False)
    print(breakdown.round(4).to_string(index=False))

    detail, group_stats = diagnostic_detail(outdir, groups, judge_frame, cfg)
    detail.to_csv(outdir / "diagnostic_detail.csv", index=False)
    save_json(outdir / "diagnostic_group_stats.json", group_stats)
    print(detail.to_string(index=False))
    print(group_stats)

    shift = transfer_shift(cfg, responses, accuracy)
    save_json(outdir / "transfer_shift.json", shift)
    print(json.dumps({ds: {k: v for k, v in s.items() if k != "example_prompt"} for ds, s in shift.items()}, indent=1))

    plot_accuracy(accuracy, figures / "accuracy_strict_vs_lenient.png")
    plot_judge(breakdown, figures / "judge_outputs.png")


if __name__ == "__main__":
    main()
