#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, math
from pathlib import Path
import argparse
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import GenerationConfig
from transformers.utils import logging as hf_logging
from tqdm.auto import tqdm

hf_logging.set_verbosity_error()

# -------------------- CONFIG --------------------
BIT4 = False
DATASETS = {
    "squad":  "../datasets/squad_v2_final/test.csv",
    "pubmed": "../datasets/pubmed_final/test.csv",
}

BASE_MODELS = {
    "qwen":  "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
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

RUN_ROOTS = {
    "squad": Path("runs_unsloth_dpp"),
    "pubmed": Path("runs_unsloth_dpp_pubmed"),
}

SQUAD_RUNS = [
    "20251008_181733_qwen_baseline_N6750_gpus2",
    "20251008_190339_qwen_semantic_N6750_gpus2",
    "20251008_195224_qwen_syntactic_N6750_gpus2",
    "20251008_209541_qwen_lexical_N6750_gpus2",
    "20251008_212719_qwen_all_N6750_gpus2",
    "20251009_020031_qwen_baseline_N13500_gpus2",
    "20251009_020130_qwen_semantic_N13500_gpus2",
    "20251009_034202_qwen_syntactic_N13500_gpus2",
    "20251009_051826_qwen_lexical_N13500_gpus2",
    "20251009_065159_qwen_all_N13500_gpus2",
    "20251009_123059_llama_baseline_N6750_gpus2",
    "20251009_134922_llama_semantic_N6750_gpus2",
    "20251009_145052_llama_syntactic_N6750_gpus2",
    "20251009_153120_llama_lexical_N6750_gpus2",
    "20251009_162316_llama_all_N6750_gpus2",
    "20251009_192950_llama_baseline_N13500_gpus2",
    "20251009_211713_llama_semantic_N13500_gpus2",
    "20251009_230226_llama_syntactic_N13500_gpus2",
    "20251010_004609_llama_lexical_N13500_gpus2",
    "20251010_023001_llama_all_N13500_gpus2",
]

PUBMED_RUNS = [
    # "20251010_143412_qwen_baseline_N11250_gpus2",
    # "20251010_163436_qwen_semantic_N11250_gpus2",
    # "20251010_183108_qwen_syntactic_N11250_gpus2",
    # "20251010_202548_qwen_lexical_N11250_gpus2",
    # "20251010_222003_qwen_all_N11250_gpus2",
    # "20251011_052933_qwen_baseline_N22500_gpus2",
    # "20251011_092817_qwen_semantic_N22500_gpus2",
    # "20251011_131937_qwen_syntactic_N22500_gpus2",
    # "20251011_170552_qwen_lexical_N22500_gpus2",
    # "20251011_205216_qwen_all_N22500_gpus2",
    "20251012_111911_llama_baseline_N11250_gpus2",
    "20251012_132515_llama_semantic_N11250_gpus2",
    "20251012_152359_llama_syntactic_N11250_gpus2",
    "20251012_172010_llama_lexical_N11250_gpus2",
    "20251012_191717_llama_all_N11250_gpus2",
    "20251013_024036_llama_baseline_N22500_gpus2",
    "20251013_064852_llama_semantic_N22500_gpus2",
    "20251013_104419_llama_syntactic_N22500_gpus2",
    "20251013_104419_llama_lexical_N22500_gpus2",
    "20251013_104419_llama_all_N22500_gpus2",
    
]

RUN_LISTS = {"squad": SQUAD_RUNS, "pubmed": PUBMED_RUNS}

OUT_ROOT = Path("inference_outputs_unsloth_new")
MAX_SEQ_LEN = 4096
BATCH_SIZE = 1
MAX_NEW_TOKENS = 64

# -------------------- HELPERS --------------------
def detect_family(run_name: str) -> str:
    return "qwen" if ("_qwen_" in run_name or run_name.startswith("qwen") or "qwen_" in run_name) else "llama"

def find_adapter_dir(run_dir: Path):
    if (run_dir / "adapter_model.safetensors").exists():
        return run_dir
    for p in sorted(run_dir.glob("checkpoint-*")) + [run_dir / "peft", run_dir / "lora"]:
        if p and (p / "adapter_model.safetensors").exists():
            return p
    return None

def find_merged_dir(run_dir: Path):
    for name in ["merged", "merged_model"]:
        cand = run_dir / name
        if (cand / "config.json").exists():
            return cand
    return None

def prep_decoder_only(model, tokenizer):
    """Ensure correct left-padding + consistent generation config for decoder-only models."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id
    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig()
    model.generation_config.pad_token_id = pad_id
    model.generation_config.eos_token_id = eos_id

def load_rows(csv_path: Path, limit: int | None):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"id": row["id"], "question": row["question"], "context": row["context"]})
            if limit is not None and len(rows) >= limit:
                break
    return rows

def build_prompts(batch, tokenizer):
    prompts = []
    for r_ in batch:
        u = USER_TEMPLATE.format(context=r_["context"], question=r_["question"])
        if hasattr(tokenizer, "apply_chat_template"):
            msgs = [{"role": "system", "content": SYS_PROMPT},
                    {"role": "user",   "content": u}
                    ]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(SYS_PROMPT + "\n\n" + u)
    return prompts

def generate_answers(model, tokenizer, rows):
    outputs = []
    num_batches = math.ceil(len(rows) / BATCH_SIZE)
    pbar = tqdm(range(0, len(rows), BATCH_SIZE), total=num_batches, leave=False, desc="batches")
    for i in pbar:
        batch = rows[i:i + BATCH_SIZE]
        prompts = build_prompts(batch, tokenizer)
        enc = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt").to(model.device)
        eos_ids = [tokenizer.eos_token_id]
        try:
            vocab = tokenizer.get_vocab()
            if "<|eot_id|>" in vocab:
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None:
                    eos_ids.append(eot_id)
        except Exception:
            pass
        
        with torch.no_grad():
            # out_ids = model.generate(
            #     **enc,
            #     max_new_tokens=MAX_NEW_TOKENS,
            #     temperature=0.8,
            #     top_p=1.0,
            #     do_sample=True,
            #     eos_token_id=tokenizer.eos_token_id,
            #     pad_token_id=tokenizer.pad_token_id,
            # )
            
            out_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,           
                top_p=0.9,
                temperature=0.7,
                eos_token_id=eos_ids,         
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                no_repeat_ngram_size=3,       
                repetition_penalty=1.1,      
                return_dict_in_generate=False
            )
            
        prompt_len = enc["input_ids"].shape[1]
        for j, ids in enumerate(out_ids):
            gen_ids = ids[prompt_len:]               
            answer = tokenizer.decode( gen_ids, skip_special_tokens=True ).strip()
            print(answer)
            outputs.append({"id": batch[j]["id"], "generated_answer": answer})
 
        pbar.set_postfix_str(f"answers={len(outputs)}")
    return outputs
       # lens = enc["attention_mask"].sum(dim=1).tolist()
        # for j, ids in enumerate(out_ids):
        #     ans_ids = ids[int(lens[j]):]
        #     ans = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
        #     ans = clean_answer(ans)
        #     outputs.append({"id": batch[j]["id"], "generated_answer": ans})
def save_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "generated_answer"])
        w.writeheader()
        w.writerows(rows)

def clean_answer(ans: str) -> str:
    if "assistant" in ans:
        parts = ans.split("assistant", 1)
        ans = parts[1]  
    ans = ans.lstrip(": \n")
    return ans.strip()

def parse_args():
    ap = argparse.ArgumentParser("Simple Unsloth inference (lean generation)")
    ap.add_argument("--dataset", choices=["squad", "pubmed", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--vanilla", action="store_true", help="Also run base (vanilla) Qwen/Llama.")
    ap.add_argument("--cuda", default="1", help="CUDA_VISIBLE_DEVICES (default: 1)")
    return ap.parse_args()

# -------------------- MAIN --------------------
def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    datasets_to_run = ["squad", "pubmed"] if args.dataset == "both" else [args.dataset]

    for ds in datasets_to_run:
        csv_path = Path(DATASETS[ds])
        if not csv_path.exists():
            print(f"[ERR] Missing dataset: {csv_path}")
            sys.exit(1)

        rows = load_rows(csv_path, args.limit)
        print(f"[INFO] Loaded {ds}: {len(rows)} rows (limit={args.limit})")

        out_dir = OUT_ROOT / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Vanilla (optional): Qwen + Llama ----------
        if args.vanilla:
            for fam_idx, fam in enumerate(tqdm(["qwen", "llama"], desc=f"{ds} • vanilla runs", leave=True), 1):
                base = BASE_MODELS[fam]
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name     = base,
                    max_seq_length = MAX_SEQ_LEN,
                    dtype          = torch.bfloat16,
                    load_in_4bit   = BIT4,
                    device_map     = "auto",
                )
                FastLanguageModel.for_inference(model)
                prep_decoder_only(model, tokenizer)

                outputs = generate_answers(model, tokenizer, rows)
                save_csv(out_dir / f"vanilla_{fam}.csv", outputs)
                print(f"[OK] {ds} [vanilla {fam_idx}/2] → vanilla_{fam}")

        # ---------- Unsloth fine-tuned runs ----------
        run_list = RUN_LISTS[ds]
        for run_idx, run_name in enumerate(tqdm(run_list, desc=f"{ds} • runs", leave=True), 1):
            run_dir = RUN_ROOTS[ds] / run_name
            fam = detect_family(run_name)
            base = BASE_MODELS[fam]

            merged = find_merged_dir(run_dir)
            if merged:
                base_or_merged = merged.as_posix()
                adapter = None
            else:
                base_or_merged = base
                adapter = find_adapter_dir(run_dir)

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name     = base_or_merged,
                max_seq_length = MAX_SEQ_LEN,
                dtype          = torch.bfloat16,
                load_in_4bit   = BIT4,
                device_map     = "auto",
            )
            FastLanguageModel.for_inference(model)
            prep_decoder_only(model, tokenizer)
            if adapter:
                model = PeftModel.from_pretrained(model, adapter.as_posix())

            outputs = generate_answers(model, tokenizer, rows)

            out_csv = out_dir / f"{run_name}.csv"
            save_csv(out_csv, outputs)
            print(f"[OK] {ds} [{run_idx}/{len(run_list)}] → {run_name} → {out_csv}")

        # ---------- Per-dataset summary ----------
        written = sum(1 for _ in out_dir.glob("*.csv"))
        print(f"[SUMMARY] {ds}: wrote {written} CSV files to {out_dir}")

    print("[DONE]")

if __name__ == "__main__":
    main()
