#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# -------------------- CONFIG --------------------
BATCH_SIZE = 64
MAX_NEW_TOKENS = 64   # adjust if needed
TEMPERATURE = 0.0     # deterministic
TOP_P = 1.0

BASE_MODELS = {
    # "qwen":  "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    # "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
    "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline",
    
}

DATASETS = {
    "squad":  "../datasets/squad_v2_final/test.csv",
    "pubmed": "../datasets/pubmed_final/test.csv",
}

SYS_PROMPT = (
    "You are a careful assistant for extractive question answering. Answer in English only."
    "Answer using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
)

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

OUT_DIR = Path("./predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- UTILS --------------------
def load_model(model_path: str):
    """Load an 8-bit quantized causal LM with left padding and a chat template-ready tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure pad token + left padding for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_8bit=True,      
        torch_dtype="auto",
    )
    return model, tokenizer

def make_messages(sys_prompt: str, user_template: str, context: str, question: str):
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_template.format(context=context, question=question)},
    ]

@torch.inference_mode()
def generate_batch(model, tokenizer, rows):
    """
    rows: list of dicts with keys: id, question, context
    Returns list of decoded strings (generated answers).
    """
    # Build chat prompts
    texts = []
    for r in rows:
        messages = make_messages(SYS_PROMPT, USER_TEMPLATE, r["context"], r["question"])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    # Tokenize (left-padded)
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min( tokenizer.model_max_length, 4096 )  # keep simple & safe
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # Generation
    eos = tokenizer.eos_token_id
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(TEMPERATURE > 0.0),
        temperature=max(1e-6, TEMPERATURE),
        top_p=TOP_P,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos,
    )

    input_lens = (enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    decoded = []
    for i in range(out.size(0)):
        gen_ids = out[i, input_lens[i]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print("*****")
        print(text)
        print("*****")
        decoded.append(text.strip())
    return decoded

def run_eval(model_key: str, model_path: str, dataset_key: str, dataset_path: str):
    model, tokenizer = load_model(model_path)
    df = pd.read_csv(dataset_path)[:50]

    results = {"id": [], "generated_answer": []}

    for col in ["id", "question", "context"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in {dataset_path}")

    rows = df[["id", "question", "context"]].to_dict(orient="records")

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc=f"{model_key}:{dataset_key}"):
        batch = rows[i:i+BATCH_SIZE]
        outs = generate_batch(model, tokenizer, batch)
        for r, gen in zip(batch, outs):
            results["id"].append(r["id"])
            results["generated_answer"].append(gen)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"preds_{model_key}_{dataset_key}_{ts}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def main():
    for mkey, mpath in BASE_MODELS.items():
        for dkey, dpath in DATASETS.items():
            if not Path(dpath).exists():
                print(f"[WARN] Skipping {dkey}: file not found at {dpath}")
                continue
            run_eval(mkey, mpath, dkey, dpath)

if __name__ == "__main__":
    main()
