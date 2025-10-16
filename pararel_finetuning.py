#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, json
from datetime import datetime, timezone
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from datasets import Dataset

# ---- Optional: CodeCarbon (rank-0 only) ----
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except Exception:
    HAS_CODECARBON = False
ENABLE_CARBON = True
CARBON_MEASURE_SECS = 1.0

# ---- Unsloth / HF / TRL ----
os.environ.setdefault("UNSLOTH_DISABLE_FUSED_LOSS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--num_samples", type=int, default=13500)
parser.add_argument("--model", type=str, default="qwen", choices=["qwen","llama"])
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--cfg", type=str,
                    choices=["baseline", "semantic", "syntactic", "lexical", "all"],
                    default="baseline")
args = parser.parse_args()

RUN_ID = os.environ.get("RUN_ID") or args.run_id or time.strftime("%Y%m%d_%H%M%S")

# ---------------- Paths ----------------
DATA_DIR = Path(".")
# DS_TRAIN = "../datasets/squad_v2_final/train.csv"
# DS_DEV   = "../datasets/squad_v2_final/dev.csv"
# AUG_CSV  = "augmented_squad_train.csv"   

DS_TRAIN = "../datasets/pubmed_final/train.csv"
DS_DEV   = "../datasets/pubmed_final/dev.csv"
AUG_CSV  = "augmented_pubmed_train.csv"     

MODELS = {
    "qwen":  "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
}
OUT_ROOT = Path("runs_unsloth_ddp_pubmed"); OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------- Hparams (2× H100) ----------------
MAX_SEQ_LEN = 4096
EPOCHS = 6
# Safe big defaults for 7B LoRA on H100 80GB @ bf16:
PER_DEVICE_TRAIN_BS = 32
GRAD_ACCUM = 4
LR = 2e-4
WARMUP_RATIO = 0.05
LOG_STEPS = 10

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

def is_main_process(trainer) -> bool:
    try:  # accelerate-backed inside HF Trainer
        return trainer.accelerator.is_main_process
    except Exception:
        return getattr(trainer, "is_world_process_zero", False)

class EpochTimerCallback(TrainerCallback):
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._epoch_t0 = None
        self.rows = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_t0 = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_t0 is None:
            return
        dur = time.time() - self._epoch_t0
        ep_float = state.epoch if state.epoch is not None else float("nan")
        ep_int = int(round(ep_float)) if isinstance(ep_float, (int, float)) else None
        self.rows.append({"epoch": ep_int, "duration_sec": dur})

    def on_train_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        # DO NOT call trainer.accelerator.wait_for_everyone() here
        if trainer is None or not (getattr(trainer, "is_world_process_zero", False) 
                                or getattr(getattr(trainer, "accelerator", None), "is_main_process", False)):
            return
        if self.rows:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_csv(self.run_dir / "epoch_times.csv", index=False)

# Unsloth fused-loss protection
class SafeLossSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if isinstance(loss, torch.Tensor):
            loss = loss.clone()
        return (loss, outputs) if return_outputs else loss

# ---------------- Data utils ----------------
def load_aug_subset(kind, num_sample): 
    df = pd.read_csv(AUG_CSV)
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
    qcol = "question" if "question" in sub.columns else "original_question"
    acol = "answer"  if "answer"  in sub.columns else "original_answer"
    ccol = "context"
    for c in [qcol, acol, ccol]:
        if c not in sub.columns:
            raise ValueError(f"Aug CSV must contain '{c}' for subset building.")
    # normalize to baseline schema
    out = pd.DataFrame({
        "id": sub.get("aug_id", pd.Series(np.arange(len(sub))).astype(str)),
        "title": sub.get("id", "aug").astype(str),
        "context": sub[ccol],
        "question": sub[qcol],
        "answer": sub[acol].fillna("unanswerable").astype(str),
    })
    return out.reset_index(drop=True).sample(n=num_sample, random_state=42)

def to_chat_text(tokenizer, ctx, q, a):
    a = "unanswerable" if (a is None or (isinstance(a, float) and np.isnan(a))) else str(a)
    messages = [
        {"role": "system", "content": "You are a helpful assistant for extractive QA."},
        {"role": "user", "content": f"Answer strictly from the context. If not present, reply exactly: unanswerable.\n\nContext:\n{ctx}\n\nQuestion:\n{q}"},
        {"role": "assistant", "content": a},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def df_to_sft_dataset(df, tokenizer):
    def _s(x): 
        return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    texts = [to_chat_text(tokenizer, _s(r["context"]), _s(r["question"]), r.get("answer"))
             for _, r in df.iterrows()]
    return Dataset.from_dict({"text": texts})

# ---------------- Model load ----------------
def load_model_tokenizer(path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
        use_gradient_checkpointing="unsloth",
        device_map=None,          
    )
    target = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=target,
        use_rslora=False,
        loftq_config=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    return model, tokenizer

# ---------------- Train ----------------
def main():
    N = args.num_samples  

    base = pd.read_csv(DS_TRAIN).sample(n=N, random_state=42)
    dev  = pd.read_csv(DS_DEV).sample(n=min(5_000, max(1, int(N * 0.15))), random_state=42)

    TRAIN_BUILDERS = {
        "baseline":  lambda: base,
        "semantic":  lambda: load_aug_subset("SEMANTIC",  N),
        "syntactic": lambda: load_aug_subset("SYNTACTIC", N),
        "lexical":   lambda: load_aug_subset("LEXICAL",   N),
        "all":       lambda: pd.concat([
                            base,
                            load_aug_subset("SEMANTIC",  N),
                            load_aug_subset("SYNTACTIC", N),
                            load_aug_subset("LEXICAL",   N),
                        ], ignore_index=True),
    }

    train_df = TRAIN_BUILDERS[args.cfg]()
    dev_df   = dev   
    
    model_key = args.model
    run_name = f"{RUN_ID}_{model_key}_{args.cfg}_N{N}_gpus{WORLD_SIZE}"
    run_dir = OUT_ROOT / run_name
    if LOCAL_RANK == 0:
        run_dir.mkdir(parents=True, exist_ok=True)

    model_path = MODELS[model_key]
    model, tokenizer = load_model_tokenizer(model_path)

    train_ds = df_to_sft_dataset(train_df, tokenizer)
    dev_ds   = df_to_sft_dataset(dev_df, tokenizer)

    def _tok(ex):
        return tokenizer(ex["text"], padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    train_tok = train_ds.map(_tok, batched=True, remove_columns=["text"])
    dev_tok   = dev_ds.map(_tok,   batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # CodeCarbon on rank 0 only
    tracker = None
    if ENABLE_CARBON and HAS_CODECARBON and (LOCAL_RANK == 0):
        tracker = EmissionsTracker(
            project_name=f"{Path(model_path).name}",
            output_dir=str(run_dir),
            output_file=f"emissions_{run_name}.csv",
            measure_power_secs=CARBON_MEASURE_SECS,
            log_level="error",
            save_to_file=True,
            gpu_ids=None,
            experiment_id=str(RUN_ID),
        )
        tracker.start()

    t0 = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    args_tr = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=max(1, PER_DEVICE_TRAIN_BS * 2),
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",

        # Precision
        bf16=torch.cuda.is_available(),
        tf32=True,

        # Logging/eval/save
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_on_each_node=False,

        # DDP
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_timeout=3600,
        
        dataloader_num_workers=2,

        # Misc
        remove_unused_columns=False,
        group_by_length=True,
        seed=42,
        run_name=run_name,
        save_safetensors=True,
    )

    trainer = SafeLossSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        args=args_tr,
        data_collator=collator,
    )

    trainer.add_callback(EpochTimerCallback(run_dir))
    trainer.train()

    ended_at = datetime.now(timezone.utc).isoformat()
    runtime_sec = time.time() - t0

    # --------- ARTIFACTS (rank-0 only) ---------
    if is_main_process(trainer):
        emissions_kg = None
        if tracker is not None:
            try:
                emissions_kg = tracker.stop()
            except Exception:
                emissions_kg = None

        # Save weights & tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(run_dir)

        # Logs
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
                compact.append({
                    "epoch": cur_epoch,
                    "train_loss": last_train_loss,
                    "eval_loss": row["eval_loss"],
                })
        if compact:
            pd.DataFrame(compact).to_csv(run_dir / "epoch_losses.csv", index=False)

        meta = {
            "run_dir": str(run_dir),
            "model_path": model_path,
            "model_key": model_key,
            "world_size": WORLD_SIZE,
            "local_rank": LOCAL_RANK,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "runtime_sec": runtime_sec,
            "epochs": EPOCHS,
            "batch_per_device": PER_DEVICE_TRAIN_BS,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": PER_DEVICE_TRAIN_BS * WORLD_SIZE * GRAD_ACCUM,
            "learning_rate": LR,
            "max_seq_len": MAX_SEQ_LEN,
            "enable_carbon": ENABLE_CARBON and HAS_CODECARBON,
            "carbon_measure_secs": CARBON_MEASURE_SECS,
            "emissions_kg": emissions_kg,
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        pd.DataFrame([meta]).to_csv(run_dir / "run_meta.csv", index=False)

if __name__ == "__main__":
    main()
