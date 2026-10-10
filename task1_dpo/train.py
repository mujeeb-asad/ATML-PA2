from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import count_parameters, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    #turns a list of preference rows into two padded batches, one for the chosen responses and one for the rejected.
    #both share the exact same prompt tokens, so the only thing that differs between the two halves is the response
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prompt_fits(tokenizer, row, max_length):
    #encode_prompt_response refuses a prompt that alone fills max_sequence_length (768), since there is then no room left
    #for any response token. those few pairs (~4% of each file) are dropped, and their ids are saved so the exact set
    #of pairs used stays on record
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
    )
    return len(prompt_ids) < max_length


def filter_rows(tokenizer, rows, max_length):
    kept = [row for row in rows if prompt_fits(tokenizer, row, max_length)]
    dropped = [row["prompt_id"] for row in rows if not prompt_fits(tokenizer, row, max_length)]
    return kept, dropped


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def pair_logps(model, chosen, rejected):
    #summed response token log probs for both halves of the pair, under whatever weights the model has active.
    #response_sequence_logprobs shifts the logits by one (token t is predicted from tokens < t) and multiplies by the
    #response mask, so prompt and padding tokens contribute nothing to the sum
    chosen_logp, _, _ = response_sequence_logprobs(model, chosen)
    rejected_logp, _, _ = response_sequence_logprobs(model, rejected)
    return chosen_logp, rejected_logp


def reference_pair_logps(model, chosen, rejected):
    #the frozen reference is the same base model with the lora adapter switched off, so no second 1.5b copy is held in
    #memory. since lora starts with B = 0 the policy and the reference are identical at step 0, exactly as dpo assumes
    with torch.no_grad(), reference_mode(model):
        return pair_logps(model, chosen, rejected)


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    tokenizer = load_tokenizer(cfg["base_model"])

    #filter first, then cut to max_examples, so every short beta fork trains on exactly the same 600 usable pairs
    rows, dropped = filter_rows(tokenizer, read_jsonl(path), int(cfg["max_sequence_length"]))
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    #fresh_lora -> the base qwen2.5-1.5b-instruct with a new r=8 adapter on q_proj/v_proj, the common initialization
    #every dpo condition starts from. the base weights stay in fp16 and frozen, peft keeps the small adapter in fp32,
    #so plain fp32 optimizer updates are used and no mixed precision machinery is needed
    model = load_policy(cfg, trainable=True, fresh_lora=True)

    #a dedicated seeded generator fixes the shuffle order, so runs that only differ in beta see the batches in the same order
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_prompt_ids": dropped,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_log_{run_name}.jsonl"
    log_path.unlink(missing_ok=True) #a rerun overwrites its own log instead of appending to the old one

    device = next(model.parameters()).device
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    #batch 2 x 8 accumulation -> each optimizer step averages 16 pairs. the last step of an epoch can see fewer
    micro_per_epoch = len(loader)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)
    total, trainable = count_parameters(model)
    print(f"[{run_name}] beta {beta}  pairs {len(bundle['rows'])}  dropped {len(bundle['dropped_prompt_ids'])}  "
          f"optimizer steps {steps_per_epoch * epochs}  trainable params {trainable:,} of {total:,}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()

    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        #running sums over the micro batches of one optimizer step, logged once the step is taken
        window = {"loss": 0.0, "acc": 0.0, "policy_margin": 0.0, "ref_margin": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0, "n": 0}
        for micro, (chosen, rejected) in enumerate(loader):
            chosen, rejected = to_device(chosen, device), to_device(rejected, device)

            #reference log probs first (no grad, adapter off), then the policy log probs with the adapter on
            ref_chosen, ref_rejected = reference_pair_logps(model, chosen, rejected)
            pol_chosen, pol_rejected = pair_logps(model, chosen, rejected)

            loss, stats = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta)

            #dividing by accum makes the summed gradient over the 8 micro batches equal to the gradient of their mean
            (loss / accum).backward()

            #implicit rewards  r(y) = beta * (log pi(y) - log ref(y)), the quantities dpo is pushing apart
            window["loss"] += loss.item()
            window["acc"] += stats["preference_accuracy"].item()
            window["policy_margin"] += stats["policy_margin_mean"].item()
            window["ref_margin"] += (ref_chosen - ref_rejected).mean().item()
            window["chosen_reward"] += (beta * (pol_chosen - ref_chosen)).mean().item()
            window["rejected_reward"] += (beta * (pol_rejected - ref_rejected)).mean().item()
            window["n"] += 1

            last_micro = micro + 1 == micro_per_epoch
            if (micro + 1) % accum == 0 or last_micro:
                #clip_grad_norm_ returns the norm before clipping, which is the useful thing to log
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), float(cfg["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                n = window.pop("n")
                record = {"run": run_name, "epoch": epoch, "step": step, "beta": beta,
                          "grad_norm": float(grad_norm), "elapsed_s": elapsed()}
                record.update({k: v / n for k, v in window.items()})
                append_jsonl(log_path, record)
                if step % 10 == 0 or last_micro:
                    print(f"[{run_name}] step {step:4d}  loss {record['loss']:.4f}  acc {record['acc']:.3f}  "
                          f"margin {record['policy_margin'] - record['ref_margin']:+.3f}  grad {record['grad_norm']:.3f}")
                window = {k: 0.0 for k in window} | {"n": 0}

    #only the lora adapter (a few mb) is written, the base weights are re-downloaded from the hub when it is loaded
    model.save_pretrained(str(output))

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
    summary = {
        "run": run_name,
        "dataset": bundle["dataset_path"],
        "beta": beta,
        "n_pairs": len(bundle["rows"]),
        "epochs": epochs,
        "optimizer_steps": step,
        "effective_batch": int(cfg["batch_size"]) * accum,
        "learning_rate": float(cfg["learning_rate"]),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "seed": int(cfg["seed"]),
        "lora": cfg["lora"],
        "wall_clock_s": elapsed(),
        "peak_vram_gb": peak_vram_gb,
        "output": str(output),
        "train_prompt_ids": [row["prompt_id"] for row in bundle["rows"]], #the exact pairs used, in file order
        "dropped_prompt_ids": bundle["dropped_prompt_ids"],
    }
    save_json(results_dir / f"train_summary_{run_name}.json", summary)
    print(f"[{run_name}] done in {summary['wall_clock_s'] / 60:.1f} min, adapter saved to {output}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
