#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from openai import OpenAI

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

def to_examples(df: pd.DataFrame) -> List[Dict]:
    examples = []
    for _, row in df.iterrows():
        gold = row["answer"] or ""
        examples.append({
            "qid": row["id"],
            "question": row["question"],
            "context": row["context"],
            "golds": [gold] if gold else [],
        })
    return examples

def ask_chat(client: OpenAI, deployment_name: str, question: str, context: str,
             temperature: float, top_p: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=deployment_name,  
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_turn|>"]
    )
    
    return (resp.choices[0].message.content or "").strip()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output_pred", default="predictions_techqa_4o.jsonl")

    parser.add_argument("--deployment", default=os.getenv("OPENAI_DEPLOYMENT", "gpt-4o"))
    parser.add_argument("--endpoint",   default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key",    default=os.getenv("OPENAI_API_KEY", ""))

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Provide --api_key or set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    print("[BOOT] loading TechQA …")
    df = pd.read_csv("techqa.csv")
    df = df[df["split"] == args.split].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit)
        print(f"[Info] Limiting to {len(df)} examples for a quick test.")
    else:
        print(f"[Info] Using full dataset: {len(df)} examples")

    examples = to_examples(df)

    print("[BOOT] initializing client …")
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    print("[BOOT] running inference via Chat Completions …")
    with open(args.output_pred, "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(examples, 1):
            try:
                pred = ask_chat(
                    client,
                    deployment_name=args.deployment,
                    question=ex["question"],
                    context=ex["context"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                print(f"[WARN] API error on {ex['qid']}: {e}", file=sys.stderr)
                pred = ""

            golds = ex["golds"]
            gold_one = golds[0] if golds else ""

            fout.write(json.dumps({
                "id": ex["qid"],
                "context": ex["context"],
                "question": ex["question"],
                "predicted_answer": pred,
                "gold_answer": gold_one
            }, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == len(examples):
                print(f"[Progress] {idx}/{len(examples)}", file=sys.stderr, flush=True)

    print(f"[DONE] wrote {args.output_pred}")

if __name__ == "__main__":
    main()