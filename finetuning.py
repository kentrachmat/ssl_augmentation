#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import ast
import os, time 
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from datasets import Dataset, load_dataset 
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling

import uuid, os

RUN_ID = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:6]}"

DATA_DIR = Path(".")
DS_SQUADV2_CSV = DATA_DIR / "final/ds_squadv2.csv"     
AUG_CSV        = DATA_DIR / "final/ds_squadv2_aug.csv"     

MODELS = {
    "gemma": "/export/home/cache/hub/unsloth-gemma-3-12b-it-offline",
    "qwen":  "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
}

# Output root
OUT_ROOT = Path("runs_unsloth1")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Training config
NUM_EPOCHS = 10
SEEDS = [13, 37, 101]

MAX_SEQ_LEN = 2048
BATCH_PER_DEVICE = 8       
GRAD_ACCUM = 4               
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10

def extract_first_text(s):
    if isinstance(s, dict):
        arr = s.get("text", [])
        return arr[0] if arr else None
    if isinstance(s, str):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                arr = obj.get("text", [])
                if hasattr(arr, "tolist"):
                    arr = arr.tolist()
                return (arr[0] if arr else None)
        except Exception:
            pass
    return None


def read_csv_guessed(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)
 
def to_chat_text(tokenizer, ctx, q, a):
    a = 'unanswerable' if a is None else a
    messages = [
        {"role": "system", "content": "You are a helpful assistant for extractive QA."},
        {"role": "user", "content": f"Answer strictly from the context. If not present, reply exactly: unanswerable.\n\nContext:\n{ctx}\n\nQuestion:\n{q}"},
        {"role": "assistant", "content": a},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def build_dev(ds_base_ids, dev_n = 300, seed = 1337):
    """Pick 300 items from SQuAD train that are NOT in ds_squadv2.csv by id."""
    df_train = load_dataset("/home/brachmat/phd/datasets/squad_v2", split='train')
    df_train = df_train.to_pandas()

    for col in ["id","title","context","question","answers"]:
        if col not in df_train.columns:
            raise ValueError(f"SQuAD training CSV must have column '{col}'")
    df_train = df_train[~df_train["id"].astype(str).isin({str(x) for x in ds_base_ids})].copy()
    if len(df_train) < dev_n:
        raise ValueError(f"Not enough non-overlapping items to sample {dev_n}. Only {len(df_train)} available.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df_train), size=dev_n, replace=False)
    dev = df_train.iloc[idx].copy().reset_index(drop=True)
   
    dev["answer"] = dev["answers"].apply(
         lambda x: x.get("text", [None])[0] if isinstance(x, dict) and len(x.get("text", [])) > 0 else None
    )
    return dev[["id","title","context","question","answer"]]

def load_baseline_train():
    df = read_csv_guessed(DS_SQUADV2_CSV).copy()
    need = ["id","title","context","question","answers"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"ds_squadv2.csv missing column '{c}'")

    df["answer"] = df["answers"].apply(extract_first_text)
    return df[["id","title","context","question","answer"]].copy()

def load_aug_subset(kind): 
    df = read_csv_guessed(AUG_CSV).copy()
    # pick label column
    type_col = None
    if "type" in df.columns:
        type_col = "type"
    elif "structure_type" in df.columns:
        type_col = "structure_type"
    else:
        raise ValueError("Aug CSV needs 'type' or 'structure_type' column.")
    # normalize
    kind = kind.upper().strip()
    df["_type_norm"] = df[type_col].astype(str).str.upper().str.strip()
    sub = df[df["_type_norm"] == kind].copy()
    if sub.empty:
        raise ValueError(f"No rows found with type == {kind} in {AUG_CSV}")

    # prefer 'question','answer','context' columns; fallback to orig_*
    qcol = "question" if "question" in sub.columns else "orig_question"
    acol = "answer"  if "answer"  in sub.columns else "orig_answer"
    ccol = "context"
    for c in [qcol, acol, ccol]:
        if c not in sub.columns:
            raise ValueError(f"Aug CSV must contain '{c}' for subset building.")
    # normalize to baseline schema
    out = pd.DataFrame({
        "id": sub.get("aug_id", pd.Series(np.arange(len(sub))).astype(str)),
        "title": sub.get("orig_id", "aug").astype(str),
        "context": sub[ccol],
        "question": sub[qcol],
        "answer": sub[acol].fillna("unanswerable").astype(str),
    })
    return out.reset_index(drop=True)

def df_to_sft_dataset(df, tokenizer):
    def _safe_str(x):
        return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)

    texts = []
    for _, r in df.iterrows():
        ctx = _safe_str(r["context"])
        q   = _safe_str(r["question"])
        a   = r["answer"]
        a   = "unanswerable" if a is None or (isinstance(a, float) and np.isnan(a)) else str(a)
        txt = to_chat_text(tokenizer, ctx, q, a)  
        texts.append(txt)

    return Dataset.from_dict({"text": texts})


def lora_targets_for(model_name_or_path: str):
    name = model_name_or_path.lower()
    if "llama" in name or "qwen" in name or "mistral" in name:
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    if "gemma" in name:
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]


def load_model_tokenizer(path):
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=path,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,            
    dtype=None,                 
    trust_remote_code=True,
    use_gradient_checkpointing="unsloth",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # LoRA 
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, lora_alpha = 16, lora_dropout = 0.05,
        target_modules = lora_targets_for(path),
        use_rslora = False,
        loftq_config = None,
    )
    model.config.use_cache = False
    return model, tokenizer

def train_one(model_path: str,
              train_df: pd.DataFrame,
              dev_df: pd.DataFrame,
              seed: int,
              run_dir: Path):

    run_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_tokenizer(model_path)
    train_ds = df_to_sft_dataset(train_df, tokenizer)
    dev_ds   = df_to_sft_dataset(dev_df, tokenizer)
    
    def _tok(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,            
            max_length=MAX_SEQ_LEN,
            return_attention_mask=True,
        )

    train_tok = train_ds.map(_tok, batched=True, remove_columns=["text"])
    dev_tok   = dev_ds.map(_tok,   batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=False,

        # core loop
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_PER_DEVICE,
        per_device_eval_batch_size=max(1, BATCH_PER_DEVICE * 2),
        gradient_accumulation_steps=GRAD_ACCUM,

        # optimizer & schedule
        learning_rate=LEARNING_RATE,      
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,        
        max_grad_norm=1.0,
        optim="adamw_torch_fused",

        # precision
        bf16=torch.cuda.is_available(),  
        fp16=False,
        tf32=True,                        

        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        eval_strategy="epoch",   
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,
        report_to="none",

        # dataloader
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,      
        group_by_length=True,    
        
        # misc
        seed=seed,
        run_name=str(run_dir.name),
        save_safetensors=True,
        
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        args=args,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()         
    tokenizer.save_pretrained(run_dir)

    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(run_dir / "log_history.csv", index=False)

    compact = []
    cur_epoch = None
    last_train_loss = None
    for row in trainer.state.log_history:
        if "epoch" in row:
            cur_epoch = int(round(row["epoch"]))
        if "loss" in row:
            last_train_loss = row["loss"]
        if "eval_loss" in row:
            compact.append({"epoch": cur_epoch, "train_loss": last_train_loss, "eval_loss": row["eval_loss"]})
    if compact:
        pd.DataFrame(compact).to_csv(run_dir / "epoch_losses.csv", index=False)

    return run_dir

def main():
    base = load_baseline_train()
    base_ids = set(base["id"].astype(str).tolist())
    dev = build_dev(base_ids, dev_n=300, seed=1337)
    dev_path = OUT_ROOT / "dev_fixed_300.csv"
    dev.to_csv(dev_path, index=False)
    print(f"[INFO] Dev set saved -> {dev_path}  (size={len(dev)})")

    TRAIN_BUILDERS = {
        "baseline":   lambda: base,
        "semantic":   lambda: load_aug_subset("SEMANTIC"),
        "syntactic":  lambda: load_aug_subset("SYNTACTIC"),
        "lexical":    lambda: load_aug_subset("LEXICAL"),
        "all":        lambda: pd.concat([
                            base,
                            load_aug_subset("SEMANTIC"),
                            load_aug_subset("SYNTACTIC"),
                            load_aug_subset("LEXICAL"),
                        ], ignore_index=True),
        }

    # Iterate models and configs
    all_groups= [] 

    for model_key, model_path in MODELS.items():
        for cfg_name, builder in TRAIN_BUILDERS.items():
            print(f"\n=== {model_key} | {cfg_name} ===")
            train_df = builder()
            print(f"[INFO] Train size: {len(train_df)}")

            run_dirs = []
            for seed in SEEDS:
                run_dir = OUT_ROOT / f"{RUN_ID}_{model_key}_{cfg_name}_seed{seed}"

                print(f"[RUN] {run_dir}")
                train_one( 
                    model_path = model_path,
                    train_df = train_df,
                    dev_df = dev,
                    seed = seed,
                    run_dir = run_dir,
                )
                run_dirs.append(run_dir)

            all_groups.append((model_key, cfg_name, run_dirs))

if __name__ == "__main__":
    main()