#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dotenv import load_dotenv
load_dotenv()

import os, sys, json, time, argparse, re
from typing import Dict, Any

import json, re, time, argparse, sys
from typing import Any, Dict, List, Optional
from statistics import mean


import pandas as pd

from openai import OpenAI

DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")
BASE_URL   = os.getenv("OPENAI_BASE_URL", "")
API_KEY    = os.getenv("OPENAI_API_KEY", "")

if not API_KEY:
    print("[Error] OPENAI_API_KEY is empty.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=BASE_URL if BASE_URL else None, api_key=API_KEY)
 

PROMPT_SHARED_HEADER = """You are an expert judge for augmented QA validity. Decide if a (context, question, answer) triple is VALID (1) or INVALID (0).

General rules:
- Grounding: The question must be answerable strictly from the provided context OR explicitly unanswerable. Never use outside knowledge.
- Answer faithfulness:
  • If the question is answerable, the provided answer must be fully supported by the context.
  • If the question is not answerable from the context, the provided answer must be exactly "unanswerable" (case-insensitive) or empty string
- Ambiguity: If the question is underspecified or allows multiple incompatible answers, mark INVALID (AMBIGUOUS).
- Out-of-scope: If the question asks about something not covered by the context, mark INVALID (OUT_OF_SCOPE).
- Span issues: If the provided answer is partially correct or omits required elements present in the context, mark INVALID (SPAN_MISSING).
- Other: Use OTHER for malformed content or edge cases not above.
"""

PROMPT_SEMANTIC = (
    PROMPT_SHARED_HEADER
    + """
Type-specific notes (Semantic):
- The question should probe a different facet or piece of information in the same context (not merely a paraphrase).
- Any grounded, correct answer consistent with the new facet is acceptable.
- If the question targets content not present in the context, mark INVALID (OUT_OF_SCOPE).

Output strict JSON only:
{
  "valid": 1 or 0,
  "reason": "SPAN_MISSING|HALLUCINATION|AMBIGUOUS|OUT_OF_SCOPE|DUPLICATE|OTHER|''",
  "notes": "short justification ( < 2 sentences)"
}
Return nothing except this JSON.
""".strip()
)

PROMPT_SYNTACTIC = (
    PROMPT_SHARED_HEADER
    + """
Type-specific notes (Syntactic):
- The question should preserve the exact meaning of an answerable query implied by the context but vary grammatical structure (e.g., active↔passive, clause reordering, simple↔complex).
- If the rewording changes the meaning or target, mark INVALID (OUT_OF_SCOPE or AMBIGUOUS as appropriate).

Output strict JSON only:
{
  "valid": 1 or 0,
  "reason": "SPAN_MISSING|HALLUCINATION|AMBIGUOUS|OUT_OF_SCOPE|DUPLICATE|OTHER|''",
  "notes": "short justification ( < 2 sentences)"
}
Return nothing except this JSON.
""".strip()
)

PROMPT_LEXICAL = (
    PROMPT_SHARED_HEADER
    + """
Type-specific notes (Lexical):
- The question should preserve the exact meaning but change surface wording (synonyms, phrasing) without altering the information requested.
- If the rephrasing introduces or removes meaning, mark INVALID (AMBIGUOUS or OUT_OF_SCOPE).

Output strict JSON only:
{
  "valid": 1 or 0,
  "reason": "SPAN_MISSING|HALLUCINATION|AMBIGUOUS|OUT_OF_SCOPE|DUPLICATE|OTHER|''",
  "notes": "short justification ( < 2 sentences)"
}
Return nothing except this JSON.
""".strip()
)

def get_system_prompt(t: str) -> str:
    t_norm = (t or "").strip().lower()
    if t_norm == "semantic":
        return PROMPT_SEMANTIC
    if t_norm == "syntactic":
        return PROMPT_SYNTACTIC
    if t_norm == "LEXICAL":
        return PROMPT_LEXICAL 
    return PROMPT_SEMANTIC

USER_TMPL = """TYPE: {type}
CONTEXT:
{context}

QUESTION:
{question}

PROVIDED_ANSWER:
{answer}
""".strip()

VALID_REASONS = {"", "SPAN_MISSING", "HALLUCINATION", "AMBIGUOUS", "OUT_OF_SCOPE", "DUPLICATE", "OTHER"}


def extract_first_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    start = s.find("{"); end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in response.")
    return json.loads(s[start:end + 1])

def normalize_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "valid" not in obj or obj["valid"] not in [0, 1, "0", "1"]:
        raise ValueError("Missing/invalid 'valid' field.")
    valid = int(obj["valid"])
    reason = str(obj.get("reason", "")).strip()
    if reason not in VALID_REASONS:
        reason = "OTHER" if valid == 0 else ""
    notes = str(obj.get("notes", "")).strip()
    return {"valid": valid, "reason": reason, "notes": notes[:600]}

def force_json(response_text: str) -> Dict[str, Any]:
    data = extract_first_json(response_text)
    return normalize_schema(data)

def _extract_logprob_payload(choice) -> Dict[str, Any]:
    """
    Returns:
      {
        "tokens": [{"token": str, "logprob": float, "top": [{"token": str, "logprob": float}, ...]} ...],
        "avg_logprob": float,
        "sum_logprob": float,
        "num_tokens": int
      }
    """
    lp = getattr(choice, "logprobs", None)
    tokens = []
    if lp and getattr(lp, "content", None):
        for step in lp.content: 
            top = []
            if getattr(step, "top_logprobs", None):
                for alt in step.top_logprobs:
                    top.append({
                        "token": getattr(alt, "token", None),
                        "logprob": float(getattr(alt, "logprob", None)),
                    })
            tokens.append({
                "token": getattr(step, "token", None),
                "logprob": float(getattr(step, "logprob", None)),
                "top": top,
            })

    vals = [t["logprob"] for t in tokens if t.get("logprob") is not None]
    return {
        "tokens": tokens,
        "avg_logprob": (mean(vals) if vals else None),
        "sum_logprob": (sum(vals) if vals else None),
        "num_tokens": len(tokens),
    }

def judge_once(system_prompt: str, user_msg: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=0.4,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        logprobs=True,         
        top_logprobs=5,         
    )
    
    choice = resp.choices[0]
    result = force_json(choice.message.content)

    lp_payload = _extract_logprob_payload(choice)
    result["avg_logprob"] = lp_payload["avg_logprob"]
    result["sum_logprob"] = lp_payload["sum_logprob"]
    result["num_tokens"] = lp_payload["num_tokens"] 
    
    result["logprobs_json"] = json.dumps(lp_payload["tokens"], ensure_ascii=False, separators=(",", ":"))
    return result

def call_judge(row_type: str, context: str, question: str, answer: str, max_retries: int = 3) -> Dict[str, Any]:
    system_prompt = get_system_prompt(row_type)
    user_msg = USER_TMPL.format(type=row_type or "", context=context or "", question=question or "", answer=answer or "")
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return judge_once(system_prompt, user_msg)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)
    return {
        "valid": 0, "reason": "OTHER",
        "notes": f"Parsing/LLM error: {type(last_err).__name__}: {last_err}",
        "avg_logprob": None, "sum_logprob": None, "num_tokens": 0, "logprobs_json": "[]"
    }

def main():
    ap = argparse.ArgumentParser(description="LLM-as-Judge for augmented QA validity (Semantic/Syntactic/Lexical).")
    ap.add_argument("--input", required=True, help="Input CSV with columns: type, context, question, answer")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of rows processed")
    ap.add_argument("--name", type=str)
    ap.add_argument("--dataset", type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = ["type", "context", "question", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[Error] Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    val_list, reason_list, notes_list = [], [], []
    avg_lp_list, sum_lp_list, n_tok_list, lp_json_list = [], [], [], []

    for i, row in df.iterrows():
        t = str(row.get("type", "")).strip()
        context = "" if pd.isna(row.get("context")) else str(row.get("context"))
        question = "" if pd.isna(row.get("question")) else str(row.get("question"))

        if pd.isna(row.get("answer")) or str(row.get("answer")).strip() == "":
            answer = "" if pd.isna(row.get("original_answer")) else str(row.get("original_answer"))
        else:
            answer = str(row.get("answer"))

        out = call_judge(t, context, question, answer)
        val_list.append(int(out["valid"]))
        reason_list.append(out["reason"])
        notes_list.append(out["notes"])
        avg_lp_list.append(out.get("avg_logprob"))
        sum_lp_list.append(out.get("sum_logprob"))
        n_tok_list.append(out.get("num_tokens"))
        lp_json_list.append(out.get("logprobs_json", "[]"))

        if (i + 1) % 25 == 0:
            print(f"[Info] Judged {i + 1}/{len(df)} items...", file=sys.stderr)

    df["valid"] = val_list
    df["reason"] = reason_list
    df["notes"] = notes_list
    df["avg_logprob"] = avg_lp_list
    df["sum_logprob"] = sum_lp_list
    df["num_tokens"] = n_tok_list
    df["logprobs_json"] = lp_json_list  

    out_df = pd.DataFrame({
        "item_id": df["aug_id"],
        "annotator": args.name,
        "valid": df["valid"],
        "reason": df["reason"],
        "notes": df["notes"],
        "avg_logprob": df["avg_logprob"],
        "sum_logprob": df["sum_logprob"],
        "num_tokens": df["num_tokens"],
        "logprobs_json": df["logprobs_json"],
    })

    out_df.to_csv(f"results/annotator_{args.dataset}_{args.name}.csv", index=False)

if __name__ == "__main__":
    main()