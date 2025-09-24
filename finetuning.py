#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, csv
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

import os
import csv
import json
import math
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


# =========================
# HARD-CODED CONFIG
# =========================

# Base model paths
MODELS = {
    "qwen":  "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
}

# Data paths
AUG_TRAIN_CSV = "final/ds_techqa_aug.csv"
RAW_CSV       = "raw/techqa.csv"          

# Variants to run
VARIANTS = ["lexical", "semantic", "syntactic", "none"]

# Output root
OUT_ROOT = "checkpoints_techqa"

# Prompts (exact)
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

# Training hyperparams (overnight-friendly)
SEED = 42
MAX_LEN = 4096
PER_DEVICE_TRAIN_BSZ = 1
PER_DEVICE_EVAL_BSZ = 1
GRAD_ACCUM_STEPS = 16
LR = 2e-4
NUM_EPOCHS = 4.0
SAVE_STEPS = 200
LOG_STEPS = 20

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Misc
ANSWER_TAG = "Answer:"


# =========================
# UTILS
# =========================

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_csv_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows

def norm(s: str) -> str:
    return (s or "").strip()

def pick_aug_answer(answer: str, orig_answer: str) -> str:
    a = norm(answer)
    if a == "":
        a = norm(orig_answer)
    # ensure non-empty label to avoid zero-length losses
    if a == "":
        a = "unanswerable"
    return a

def pick_raw_answer(answer: str) -> str:
    a = norm(answer)
    if a == "":
        a = "unanswerable"
    return a

def is_aug_row_for_variant(row_type: str, variant: str) -> bool:
    t = norm(row_type).lower()
    if variant == "lexical":
        return t == "lexical"
    if variant == "semantic":
        return t == "semantic"
    if variant == "syntactic":
        return t in {"syntactic", "syntatic"}   # tolerate typo
    return False


# =========================
# DATA LOADING
# =========================

def load_aug_train_variant(variant: str) -> List[Dict]:
    """From augmented CSV, filter by type for lexical/semantic/syntactic."""
    assert variant in {"lexical", "semantic", "syntactic"}
    rows = read_csv_rows(AUG_TRAIN_CSV)
    out = []
    for r in rows:
        if is_aug_row_for_variant(r.get("type", ""), variant):
            ctx = norm(r.get("context", ""))
            q   = norm(r.get("question", ""))
            a   = pick_aug_answer(r.get("answer", ""), r.get("orig_answer", ""))
            if ctx and q and a:
                out.append({"context": ctx, "question": q, "answer": a})
    if not out:
        raise RuntimeError(f"No training rows found in {AUG_TRAIN_CSV} for variant={variant}")
    return out

def load_raw_split(split_name: str) -> List[Dict]:
    """From raw CSV, filter by split (train/dev)."""
    split_name = split_name.lower()
    rows = read_csv_rows(RAW_CSV)
    out = []
    for r in rows:
        if norm(r.get("split", "")).lower() == split_name:
            ctx = norm(r.get("context", ""))
            q   = norm(r.get("question", ""))
            a   = pick_raw_answer(r.get("answer", ""))
            if ctx and q and a is not None:
                out.append({"context": ctx, "question": q, "answer": a})
    if not out:
        raise RuntimeError(f"No rows found in {RAW_CSV} for split={split_name}")
    return out


# =========================
# DATASET + COLLATOR
# =========================

class TechQADataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.sys_prefix = f"System: {SYS_PROMPT}\n\n"
        self.answer_tag_ids = self.tok(ANSWER_TAG, add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        user = USER_TEMPLATE.format(context=r["context"], question=r["question"])
        text = self.sys_prefix + user + r["answer"]

        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )

        input_ids = enc["input_ids"]
        labels = input_ids.copy()

        # Mask everything up to and including "Answer:" so loss is on answer tokens only
        start = 0
        L = len(self.answer_tag_ids)
        if L > 0:
            for i in range(0, len(input_ids) - L + 1):
                if input_ids[i: i+L] == self.answer_tag_ids:
                    start = i + L
                    break
        for i in range(start):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

class PadCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            attn.append(x["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# =========================
# TRAINING
# =========================

def train_one(model_key: str, model_path: str, variant: str):
    # ----- data -----
    if variant == "none":
        train_rows = load_raw_split("train")      # none == raw train
    else:
        train_rows = load_aug_train_variant(variant)

    dev_rows = load_raw_split("dev")

    out_dir = os.path.join(OUT_ROOT, f"{model_key}_{variant}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] {model_key}/{variant}: #train={len(train_rows)}  #dev={len(dev_rows)}  -> {out_dir}")

    # ----- tokenizer -----
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ----- datasets -----
    train_ds = TechQADataset(train_rows, tok, MAX_LEN)
    dev_ds   = TechQADataset(dev_rows, tok, MAX_LEN)
    collator = PadCollator(tok)

    # ----- model + LoRA -----
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False  # needed with grad checkpointing

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # ----- training args -----
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        seed=SEED,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BSZ,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BSZ,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=LOG_STEPS,
        logging_first_step=True,
        report_to=["tensorboard"],

        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        disable_tqdm=False,  # show progress bars in terminal
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    print(f"[INFO] Starting training: {out_dir}")
    trainer.train(resume_from_checkpoint=True)  # auto-resume if a checkpoint exists

    # Save final adapter + loss history
    trainer.save_model(out_dir)
    with open(os.path.join(out_dir, "loss_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"[OK] saved -> {out_dir}")


def main():
    set_seed(SEED)
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # If you want to pin GPUs manually, uncomment:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    for model_key, model_path in MODELS.items():
        for variant in VARIANTS:
            train_one(model_key, model_path, variant)


if __name__ == "__main__":
    main()