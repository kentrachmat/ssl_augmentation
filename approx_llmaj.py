#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


import json
import sys
from typing import Dict, Any, List

import pandas as pd
from vllm import LLM, SamplingParams

# --------------------------
# HARD-CODED CONFIG
# --------------------------
MODEL_PATH = "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline"
INPUT_JSONL = "results/predictions_techqa_llama.jsonl"   # one JSON object per line
OUTPUT_CSV  = "judge_results.csv"

TENSOR_PARALLEL_SIZE = 2
DTYPE = "bfloat16"  # "auto", "float16", or "bfloat16"

SYSTEM_PROMPT = (
    "You are a careful, impartial LLM-as-Judge that decides whether a proposed answer "
    "is grounded strictly in the provided context (no external knowledge).\n\n"
    "Task:\n"
    "Given CONTEXT, QUESTION, and PROPOSED_ANSWER, decide:\n"
    "  A = YES, the answer is fully supported by the context.\n"
    "  B = NO, the answer is not supported / contradicts / missing from the context.\n"
    "  C = UNSURE, the context is ambiguous or insufficient to decide.\n\n"
    "Rules:\n"
    "- If context clearly supports the answer (contiguous textual support is present), choose A.\n"
    "- If support is absent or contradicted by context, choose B (HALLUCINATION / OUT_OF_SCOPE / SPAN_MISSING).\n"
    "- If the question is underspecified or multiple incompatible answers fit, choose C (AMBIGUOUS).\n"
    "- Never use outside knowledge.\n"
    "- Briefly state your reasoning, then output STRICT JSON only.\n\n"
    "Output format (strict JSON only; no code fences):\n"
    "{\n"
    '  "label": "A|B|C",\n'
    '  "reason": "SPAN_MISSING|HALLUCINATION|OUT_OF_SCOPE|AMBIGUOUS|OTHER|\'\'",\n'
    '  "notes": "one short sentence"\n'
    "}\n"
    "Do not include anything except this JSON after your reasoning."
)

USER_TMPL = (
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "PROPOSED_ANSWER:\n{answer}\n\n"
    "First, think briefly. Then on a new line, start with exactly one of A, B, or C, followed by a newline, "
    "then the strict JSON described above."
)

LABEL_MAP = {"A": "YES", "B": "NO", "C": "UNSURE"}


# --------------------------
# PROMPT BUILDING
# --------------------------
def build_chat(messages: List[Dict[str, str]]) -> str:
    """Minimal chat template for Llama 3 Instruct."""
    out = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            out.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}\n<|eot_id|>")
        elif role == "user":
            out.append(f"<|start_header_id|>user<|end_header_id|>\n{content}\n<|eot_id|>")
        elif role == "assistant":
            out.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}\n<|eot_id|>")
    out.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(out)


# --------------------------
# OUTPUT PARSING
# --------------------------
def parse_label_and_json(text: str) -> Dict[str, Any]:
    """
    Expect model output like:
      A
      { "label":"A", "reason":"...", "notes":"..." }
    """
    s = text.strip()
    nl = s.find("\n")
    if nl == -1:
        return {"label": "B", "reason": "OTHER", "notes": "Malformed: no newline."}
    first = s[:nl].strip()
    rest = s[nl + 1 :].strip()

    lb = first if first in ("A", "B", "C") else None

    start = rest.find("{")
    end = rest.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"label": lb or "B", "reason": "OTHER", "notes": "Malformed: no JSON."}

    try:
        data = json.loads(rest[start : end + 1])
        label = data.get("label", lb) or "B"
        if label not in ("A", "B", "C"):
            label = "B"
        reason = str(data.get("reason", "")).strip()
        notes = str(data.get("notes", "")).strip()
        return {"label": label, "reason": reason, "notes": notes}
    except Exception as e:
        return {"label": lb or "B", "reason": "OTHER", "notes": f"Bad JSON: {type(e).__name__}"}


def first_label_confidence(top_logprobs) -> Dict[str, float]:
    """
    Confidence from the probability of the *first generated token* being 'A', 'B', or 'C'.
    top_logprobs: list[dict(token->logprob)] for each generated position.
    """
    if not top_logprobs or not top_logprobs[0]:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "max_label": None, "confidence": 0.0}

    first = top_logprobs[0]

    def token_prob(prefix: str) -> float:
        # try exact or stripped variants (e.g., 'A', ' A', 'A\n')
        cands = [tok for tok in first.keys() if tok.strip().startswith(prefix)]
        if not cands:
            return 0.0
        best_logp = max(first[tok] for tok in cands)
        import math
        return float(math.exp(best_logp))

    pA, pB, pC = token_prob("A"), token_prob("B"), token_prob("C")
    max_label, max_p = "A", pA
    if pB > max_p:
        max_label, max_p = "B", pB
    if pC > max_p:
        max_label, max_p = "C", pC

    return {"A": pA, "B": pB, "C": pC, "max_label": max_label, "confidence": max_p}


# --------------------------
# CORE JUDGE
# --------------------------
def judge_rows(llm: LLM, rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompts = []
    for r in rows:
        context = r.get("context", "") or ""
        question = r.get("question", "") or ""
        answer = r.get("predicted_answer", "") or ""  # <-- use predicted_answer field
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TMPL.format(context=context, question=question, answer=answer)},
        ]
        prompts.append(build_chat(messages))

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        logprobs=5,   # needed for first-token confidence
    )

    outputs = llm.generate(prompts, sampling)
    results = []
    for r, out in zip(rows, outputs):
        seq = out.outputs[0]
        text = seq.text
        # In vLLM, when logprobs>0, you get .top_logprobs: List[Dict[str, float]]
        top_logprobs = getattr(seq, "top_logprobs", None)
        conf = first_label_confidence(top_logprobs) if top_logprobs is not None else {
            "A": 0.0, "B": 0.0, "C": 0.0, "max_label": None, "confidence": 0.0
        }
        parsed = parse_label_and_json(text)
        label = parsed["label"]

        results.append({
            "label": label,                        # 'A'/'B'/'C'
            "decision": LABEL_MAP[label],          # 'YES'/'NO'/'UNSURE'
            "reason": parsed.get("reason", ""),
            "notes": parsed.get("notes", ""),
            "pA": round(conf["A"], 6),
            "pB": round(conf["B"], 6),
            "pC": round(conf["C"], 6),
            "conf_label": conf["max_label"],
            "confidence": round(conf["confidence"], 6),
            "context": r.get("context", ""),
            "question": r.get("question", ""),
            "predicted_answer": r.get("predicted_answer", ""),
            "raw": text.strip()
        })
    return results


# --------------------------
# MAIN
# --------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    print(f"[Info] Loading model: {MODEL_PATH}", file=sys.stderr)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype=DTYPE,
        trust_remote_code=True,
    )

    print(f"[Info] Reading: {INPUT_JSONL}", file=sys.stderr)
    rows = read_jsonl(INPUT_JSONL)

    print(f"[Info] Judging {len(rows)} items...", file=sys.stderr)
    results = judge_rows(llm, rows)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Done] Saved {len(df)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
