from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.data import repo_path
from common.models import resolve_dtype

PAIRWISE_RUBRIC = """You are comparing two candidate solutions to the same math problem.

Evaluate mathematical correctness, internal consistency, relevance, and whether the stated final answer follows from the reasoning.
Do not prefer a response merely because it is longer, more confident, or more polished.
If neither response is meaningfully better, return TIE.

Return exactly one token: A, B, or TIE.

Problem:
{problem}

Candidate A:
{a}

Candidate B:
{b}

Preference:"""


class PairwiseAIJudge:
    """Fixed course judge with deterministic A/B orientation balancing and persistent caching."""

    def __init__(self, cfg: dict, cache_path: str | Path):
        self.cfg = cfg
        self.cache_path = repo_path(cache_path)
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}
        #added for the evaluation -> the judge's raw decoded text for every verdict, kept next to the cache. the verdict
        #logic below is unchanged, this only makes a genuine TIE distinguishable from unparseable output (which the
        #parser also maps to TIE), so ambiguous judge outputs can be reported separately as the manual asks
        self.raw_path = self.cache_path.with_name(self.cache_path.stem + "_raw.json")
        self.raw = json.loads(self.raw_path.read_text(encoding="utf-8")) if self.raw_path.exists() else {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["ai_judge_model"], padding_side="left", use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available() and bool(cfg.get("quantize_frozen_models", True)):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolve_dtype(cfg.get("dtype", "float16")),
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = resolve_dtype(cfg.get("dtype", "float16"))

        self.model = AutoModelForCausalLM.from_pretrained(cfg["ai_judge_model"], **kwargs)
        self.model.eval()

    def _key(self, problem, a, b):
        payload = json.dumps(
            {"model": self.cfg["ai_judge_model"], "problem": problem, "a": a, "b": b},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @torch.no_grad()
    def compare(self, problem: str, a: str, b: str):
        key = self._key(problem, a, b)
        if key in self.cache:
            return self.cache[key]

        swap = int(key[:8], 16) % 2 == 1
        aa, bb = (b, a) if swap else (a, b)
        text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(next(self.model.parameters()).device)
        out = self.model.generate(
            ids,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
        m = re.search(r"\b(A|B|TIE)\b", decoded)
        result = m.group(1) if m else "TIE"
        if swap:
            result = {"A": "B", "B": "A", "TIE": "TIE"}[result]

        self.cache[key] = result
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        self.raw[key] = decoded
        self.raw_path.write_text(json.dumps(self.raw, indent=2), encoding="utf-8")
        return result

    def output_kind(self, problem: str, a: str, b: str):
        #added -> what the judge actually wrote for this pair, next to the verdict the parser made of it ->
        #  "clean"           exactly one kind of label (A, B or TIE), the verdict is the judge's answer
        #  "unparsed"        no label at all, the TIE it was given is the parser's fallback
        #  "multiple_labels" more than one kind of label (e.g. "A, B"), the parser kept the first one
        #None when no raw output was recorded for the pair
        raw = self.raw.get(self._key(problem, a, b))
        if raw is None:
            return None
        labels = set(re.findall(r"\b(A|B|TIE)\b", raw))
        return "unparsed" if not labels else "clean" if len(labels) == 1 else "multiple_labels"

    def group_rewards(self, problem: str, responses: list[str]):
        k = len(responses)
        wins = np.zeros(k, dtype=float)
        ties = np.zeros(k, dtype=float)
        for i, j in itertools.combinations(range(k), 2):
            pref = self.compare(problem, responses[i], responses[j])
            if pref == "A":
                wins[i] += 1
            elif pref == "B":
                wins[j] += 1
            else:
                ties[i] += 1
                ties[j] += 1
        return ((wins + 0.5 * ties) / max(k - 1, 1)).tolist()
