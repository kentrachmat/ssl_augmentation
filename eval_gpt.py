#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")     
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")

TEMPERATURE = 0.5
TOP_P       = 1.0
MAX_TOKENS  = 1024
STOP_TOKENS = ["<|eot_id|>", "<|end_of_turn|>"]

# =========================
# Prompt templates
# =========================
SYS_PROMPT = (
    "You are a careful assistant for extractive question answering. "
    "Answer using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
)
USER_TEMPLATE = (
    "Answer the question strictly based on the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# =========================
# Helpers
# =========================
def ask_chat(client: OpenAI, question: str, context: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()

def load_techqa(split: str, limit: int | None) -> Tuple[pd.DataFrame, List[Dict]]:
    df = pd.read_csv("raw/techqa.csv")
    df = df[df["split"] == split].copy()
    if limit and limit > 0:
        df = df.head(limit)

    examples = []
    for _, row in df.iterrows():
        gold = row.get("answer") or ""
        examples.append({
            "id": row["id"],
            "question": row["question"],
            "context": row["context"],
            "golds": [gold] if gold else [],
        })
    return df, examples

def load_squad_v2(split: str, limit: int | None) -> Tuple[pd.DataFrame, List[Dict]]:
    ds = load_dataset("/home/brachmat/phd/datasets/squad_v2", split=split)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    df = ds.to_pandas()

    examples = []
    for _, row in df.iterrows():
        answers = row["answers"] if isinstance(row["answers"], dict) else {"text": [], "answer_start": []}
        if len(answers["text"]) > 0 and answers["text"][0].strip():
            gold_texts = answers["text"][0]
        else:
            gold_texts = "unanswerable"
            
        examples.append({
            "id": row["id"],
            "question": row["question"],
            "context": row["context"],
            "golds": gold_texts,
        })
    return df, examples

def write_techqa_output(path_jsonl: str, rows: List[Dict]) -> None:
    with open(path_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_squad_v2_outputs(path_pred_json: str, path_jsonl: str, rows: List[Dict]) -> None:
    id2pred = {r["id"]: r["predicted_answer"] for r in rows}
    with open(path_pred_json, "w", encoding="utf-8") as f:
        json.dump(id2pred, f, ensure_ascii=False, indent=2)

    with open(path_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["techqa", "squad_v2"],
                        help="Choose which dataset loader + output format to use.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0, help="Optional quick test limit.")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[ERROR] Set OPENAI_API_KEY in your environment or .env", file=sys.stderr)
        sys.exit(1)

    print(f"[BOOT] Dataset: {args.dataset} | Split: {args.split}")
    if args.dataset == "techqa":
        df, examples = load_techqa(args.split, args.limit)
        print(f"[Info] Using {len(df)} TechQA examples.")
    else:
        df, examples = load_squad_v2(args.split, args.limit)
        print(f"[Info] Using {len(df)} SQuAD v2 examples.")

    print("[BOOT] Initializing client …")
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    print("[RUN] Inference …")
    out_rows = []
    for idx, ex in enumerate(examples, 1):
        try:
            pred = ask_chat(client, question=ex["question"], context=ex["context"])
        except Exception as e:
            print(f"[WARN] API error on {ex['id']}: {e}", file=sys.stderr)
            pred = ""

        golds = ex["golds"] or []
        gold_one = golds if golds else ""

        out = {
            "id": ex["id"],
            "question": ex["question"],
            "context": ex["context"],
            "predicted_answer": pred,
            "gold_answer": gold_one,
        }
        # if golds:
        #     out["all_gold_answers"] = golds

        out_rows.append(out)

        if idx % 25 == 0 or idx == len(examples):
            print(f"[Progress] {idx}/{len(examples)}", file=sys.stderr, flush=True)

    if args.dataset == "techqa":
        out_path = "results/predictions_techqa_gpt4o.jsonl"
        write_techqa_output(out_path, out_rows)
        print(f"[DONE] Wrote {out_path}")
    else:
        out_json  = "results/predictions_squadv2_gpt4o.json"  
        out_jsonl = "results/predictions_squadv2_gpt4o.jsonl"   
        write_squad_v2_outputs(out_json, out_jsonl, out_rows)
        print(f"[DONE] Wrote {out_json} and {out_jsonl}")

if __name__ == "__main__":
    main()