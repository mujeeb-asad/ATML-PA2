from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


#the four controlled pairs of every problem -> the clean correct response against one perturbed variant. in every pair
#the clean response is the diagnostically better one, what changes is WHY the other one is worse ->
#  corrupt_reasoning_correct_final : same correct final answer, an intermediate step is wrong   (reasoning-only change)
#  persuasive_filler_correct       : same correct answer and reasoning, plus irrelevant persuasive filler
#  good_reasoning_wrong_final      : reasoning kept, the designated final answer is wrong         (outcome change)
#  gold_distractor_wrong_final     : mentions the gold number, but the designated final answer is wrong (outcome change)
PERTURBED = ["corrupt_reasoning_correct_final", "persuasive_filler_correct", "good_reasoning_wrong_final",
             "gold_distractor_wrong_final"]
REASON_VARIANT = "corrupt_reasoning_correct_final" #the pair behind S_reason
OUTCOME_VARIANTS = ["good_reasoning_wrong_final", "gold_distractor_wrong_final"] #the pairs behind S_outcome


def score_pairs(groups, judge):
    #every clean-vs-perturbed pair, scored by both feedback mechanisms. "better" = prefers the clean response
    pairs = []
    for problem_id, variants in groups.items():
        clean = variants["clean_correct"]
        gold = str(clean["gold_final"])
        clean_exact = exact_reward(clean["response"], gold)
        for variant_type in PERTURBED:
            other = variants[variant_type]
            other_exact = exact_reward(other["response"], gold)
            #rlvr -> the binary verifier can only prefer a response when the two rewards differ, equal rewards are a tie
            verifier = "better" if clean_exact > other_exact else "wrong" if other_exact > clean_exact else "tie"
            #rlaif -> the pairwise judge sees the problem and both texts, A is the clean response
            verdict = judge.compare(clean["question"], clean["response"], other["response"])
            pairs.append({
                "problem_id": problem_id,
                "variant_type": variant_type,
                "clean_exact": clean_exact,
                "variant_exact": other_exact,
                "verifier": verifier,
                "judge": {"A": "better", "TIE": "tie", "B": "wrong"}[verdict],
                "judge_output": judge.output_kind(clean["question"], clean["response"], other["response"]), #clean / unparsed / multiple_labels
                "question": clean["question"],
                "clean_response": clean["response"],
                "variant_response": other["response"],
            })
    return pairs


def preference_rates(pairs):
    #per perturbation type and per mechanism -> share of pairs where it prefers the better (clean) response, ties,
    #or prefers the worse one
    rows = []
    for variant_type in PERTURBED:
        frame = pairs[pairs["variant_type"] == variant_type]
        for mechanism in ["verifier", "judge"]:
            rows.append({
                "variant_type": variant_type,
                "mechanism": "rlvr_exact_verifier" if mechanism == "verifier" else "rlaif_pairwise_judge",
                "pairs": len(frame),
                "better_rate": (frame[mechanism] == "better").mean(),
                "tie_rate": (frame[mechanism] == "tie").mean(),
                "wrong_rate": (frame[mechanism] == "wrong").mean(),
                #ambiguous judge outputs (the verifier never has one) -> no label at all, or several labels at once
                "unparsed_rate": (frame["judge_output"] == "unparsed").mean() if mechanism == "judge" else 0.0,
                "multiple_label_rate": (frame["judge_output"] == "multiple_labels").mean() if mechanism == "judge" else 0.0,
            })
    return pd.DataFrame(rows)


def sensitivities(pairs):
    #S_reason  = Pr[R(clean) > R(reason-corrupt)]      over the pairs with the same correct final answer
    #S_outcome = Pr[R(correct final) > R(wrong final)] over the outcome-changing pairs
    #a tie counts as "not greater", so a mechanism that cannot see a difference scores 0 on it
    reason = pairs[pairs["variant_type"] == REASON_VARIANT]
    outcome = pairs[pairs["variant_type"].isin(OUTCOME_VARIANTS)]
    filler = pairs[pairs["variant_type"] == "persuasive_filler_correct"]
    rows = []
    for mechanism, label in [("verifier", "rlvr_exact_verifier"), ("judge", "rlaif_pairwise_judge")]:
        row = {
            "mechanism": label,
            "S_reason": (reason[mechanism] == "better").mean(),
            "S_reason_tie_rate": (reason[mechanism] == "tie").mean(),
            "S_reason_wrong_rate": (reason[mechanism] == "wrong").mean(),
            "S_outcome": (outcome[mechanism] == "better").mean(),
            "S_outcome_tie_rate": (outcome[mechanism] == "tie").mean(),
            "S_outcome_wrong_rate": (outcome[mechanism] == "wrong").mean(),
            #filler susceptibility -> the mechanism prefers the response padded with persuasive filler over the clean one
            "filler_preferred_rate": (filler[mechanism] == "wrong").mean(),
            "filler_tie_rate": (filler[mechanism] == "tie").mean(),
        }
        for variant_type in OUTCOME_VARIANTS: #S_outcome split by the two outcome-changing perturbations
            row[f"S_outcome_{variant_type}"] = (outcome[outcome["variant_type"] == variant_type][mechanism] == "better").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def group_rewards(groups, judge):
    #the reward each mechanism would actually hand to the five variants if they were one sampled group ->
    #rlvr gives the binary exact reward, rlaif the normalised pairwise win rate (wins + 0.5 ties) / (K - 1) over all
    #ten pairs of the group. the two scales are not comparable with each other, only the ordering inside each is
    rows = []
    for problem_id, variants in groups.items():
        order = sorted(EXPECTED_VARIANTS)
        responses = [variants[v]["response"] for v in order]
        rlaif = judge.group_rewards(variants["clean_correct"]["question"], responses)
        for variant_type, response, reward in zip(order, responses, rlaif):
            rows.append({"problem_id": problem_id, "variant_type": variant_type,
                         "rlvr_reward": exact_reward(response, str(variants[variant_type]["gold_final"])),
                         "rlaif_reward": reward})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    #no policy is involved here -> the responses are the supplied, manually validated ones, only the two feedback
    #mechanisms are run. the judge shares its on-disk cache with evaluate_math
    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")

    pairs = score_pairs(groups, judge)
    write_jsonl(outdir / "diagnostic_pairs.jsonl", pairs)
    pairs = pd.DataFrame(pairs)

    rates = preference_rates(pairs)
    rates.to_csv(outdir / "diagnostic_preference_rates.csv", index=False)
    print(rates.round(3).to_string(index=False))

    sensitivity = sensitivities(pairs)
    sensitivity.to_csv(outdir / "diagnostic_sensitivity.csv", index=False)
    print(sensitivity.set_index("mechanism").T.round(3).to_string())

    rewards = group_rewards(groups, judge)
    rewards.to_csv(outdir / "diagnostic_group_rewards.csv", index=False)
    print(rewards.groupby("variant_type")[["rlvr_reward", "rlaif_reward"]].mean().round(3).to_string())

    #the verifier against the labels the staff attached to the set, a check that the parser reads every variant as intended
    rows = read_jsonl(cfg["paths"]["task5_diagnostics"])
    matches = int(np.sum([exact_reward(r["response"], str(r["gold_final"])) == r["expected_exact_reward"] for r in rows]))
    save_json(outdir / "diagnostic_verifier_check.json", {"rows": len(rows), "verifier_matches_expected_reward": matches})
    print(f"exact verifier reproduces the staff's expected reward on {matches} of {len(rows)} diagnostic responses")


if __name__ == "__main__":
    main()
