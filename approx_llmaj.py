import os, sys, json
from typing import List, Dict, Any
import torch
import pandas as pd
from unsloth import FastLanguageModel
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_PATH = "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline"
INPUT_JSONL = "results/predictions_squadv2_llama.jsonl"
OUTPUT_CSV  = "judge_results.csv"

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

DTYPE            = torch.float16
DEVICE           = "cuda"
BATCH_SIZE       = 4
MAX_NEW_TOKENS   = 256
CTX_LEN          = 4096
AGGREGATION      = "max"  

def build_chat(messages): 
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{m['content']}\n<|eot_id|>")
        elif role == "user":
            out.append(f"<|start_header_id|>user<|end_header_id|>\n{m['content']}\n<|eot_id|>")
        elif role == "assistant":
            out.append(f"<|start_header_id|>assistant<|end_header_id|>\n{m['content']}\n<|eot_id|>")
    out.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(out)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def parse_reason_json(text: str):
    """
    Extract the first JSON object from text and return reason/notes/label (if present).
    If no JSON found, return blanks.
    """
    s = text.strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return {"label": "", "reason": "", "notes": ""}
    try:
        data = json.loads(s[i:j+1])
        return {
            "label": str(data.get("label","")).strip(),
            "reason": str(data.get("reason","")).strip(),
            "notes":  str(data.get("notes","")).strip(),
        }
    except Exception:
        return {"label": "", "reason": "", "notes": ""}

def vocab_prefix_id_sets(tokenizer): 
    id2tok = {v: k for k, v in tokenizer.get_vocab().items()}
    buckets = {"A": [], "B": [], "C": []}
    for tid in id2tok:
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        st = s.strip()
        if not st:
            continue
        for pref in ("A", "B", "C"):
            if st.startswith(pref):
                buckets[pref].append(tid)
    return buckets

def pool_probs(vec: torch.Tensor, ids: List[int], how: str = "max") -> torch.Tensor:
    """
    Pool probability for a bucket: 'max' ≈ probability of the first label token,
    'sum' = total mass of all tokens starting with that label.
    vec: [vocab_size] probabilities for one example.
    """
    if not ids:
        return torch.tensor(0.0, device=vec.device)
    sub = vec[ids]
    return sub.max() if how == "max" else sub.sum()

def judge_rows_singlepass(model, tokenizer, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each item:
      1) Build prompt (assistant turn open).
      2) One forward() to get next-token logits at the end of the prompt.
         -> Softmax -> pA/pB/pC from the first label token(s).
         -> Choose label L in {A,B,C} (argmax by pooled prob).
      3) Seed the label into the prompt ("L\\n") and do one generate() to get the JSON.
    """
    prompts: List[str] = []
    for r in rows:
        context  = r.get("context", "") or ""
        question = r.get("question", "") or ""
        answer   = r.get("predicted_answer", "") or ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TMPL.format(context=context, question=question, answer=answer)},
        ]
        prompts.append(build_chat(messages))

    # Token sets for A/B/C (once)
    abc_buckets = vocab_prefix_id_sets(tokenizer)
    abc_union: List[int] = sorted(set(abc_buckets["A"] + abc_buckets["B"] + abc_buckets["C"]))

    results= []

    for batch_rows, batch_prompts in zip(chunked(rows, BATCH_SIZE), chunked(prompts, BATCH_SIZE)):
        # 1) Tokenize prompts
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CTX_LEN,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        prompt_lens = (enc["input_ids"] != pad_id).sum(dim=1)  # [batch]

        # 2) One forward() to get first-step logits (no generation yet)
        with torch.no_grad():
            out = model(**enc, use_cache=True)
        logits = out.logits  # [batch, seq_len, vocab]
        # pick logits at the last prompt position for each element
        idx = prompt_lens - 1  # [batch]
        first_logits = logits[torch.arange(logits.size(0), device=logits.device), idx, :]  # [batch, vocab]
        first_probs = torch.softmax(first_logits, dim=-1)  # [batch, vocab]

        # Compute pooled probs pA/pB/pC and pick label + confidence
        batch_labels: List[str] = []
        batch_conf: List[float] = []
        batch_pA, batch_pB, batch_pC = [], [], []
        chosen_tok_ids: List[int] = []

        for i in range(first_probs.size(0)):
            vec = first_probs[i]
            pA = float(pool_probs(vec, abc_buckets["A"], AGGREGATION).item())
            pB = float(pool_probs(vec, abc_buckets["B"], AGGREGATION).item())
            pC = float(pool_probs(vec, abc_buckets["C"], AGGREGATION).item())
            batch_pA.append(pA); batch_pB.append(pB); batch_pC.append(pC)

            # argmax among A/B/C
            if pA >= pB and pA >= pC:
                label = "A"; conf = pA; bucket_ids = abc_buckets["A"]
            elif pB >= pA and pB >= pC:
                label = "B"; conf = pB; bucket_ids = abc_buckets["B"]
            else:
                label = "C"; conf = pC; bucket_ids = abc_buckets["C"]

            # choose the actual first label token id we would emit (top within bucket)
            if bucket_ids:
                # top token by prob inside the chosen bucket
                sub = first_probs[i][bucket_ids]
                best_local = int(torch.argmax(sub).item())
                chosen_tid = int(bucket_ids[best_local])
            else:
                # safety: fall back to global argmax
                chosen_tid = int(torch.argmax(first_probs[i]).item())

            batch_labels.append(label)
            batch_conf.append(conf)
            chosen_tok_ids.append(chosen_tid)

        # 3) Seed the chosen label token + newline into the assistant turn, then generate JSON
        #    We rebuild prompts with the label literally appended as text to the assistant prefix.
        seeded_prompts: List[str] = []
        for p, lab in zip(batch_prompts, batch_labels):
            seeded_prompts.append(p + lab + "\n")

        gen_inputs = tokenizer(
            seeded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CTX_LEN,
        )
        gen_inputs = {k: v.to(DEVICE) for k, v in gen_inputs.items()}

        with torch.no_grad():
            out_gen = model.generate(
                **gen_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=False,                    
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode ONLY the continuation beyond the seeded prompt
        if tokenizer.pad_token_id is None:
            prompt_lens_gen = gen_inputs["input_ids"].shape[1] * torch.ones(
                gen_inputs["input_ids"].shape[0], dtype=torch.long, device=gen_inputs["input_ids"].device
            )
        else:
            prompt_lens_gen = (gen_inputs["input_ids"] != (tokenizer.pad_token_id or tokenizer.eos_token_id)).sum(dim=1)

        for i, row in enumerate(batch_rows):
            seq_ids = out_gen.sequences[i]
            pl = int(prompt_lens_gen[i].item())
            gen_ids = seq_ids[pl:]  # generated continuation (ideally pure JSON)
            cont_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Build 'raw' as required: first line label + JSON block
            raw = batch_labels[i] + "\n" + cont_text.strip()

            parsed = parse_reason_json(cont_text)
            final_label = batch_labels[i]  # our seeded choice
            # If JSON includes a (conflicting) label, you can reconcile here:
            if parsed.get("label") in ("A","B","C") and parsed["label"] != final_label:
                # Keep our letter (first-token choice) as the decision; store JSON's as a note
                parsed["notes"] = (parsed.get("notes","") + f" | JSON_label={parsed['label']}").strip()
                parsed["label"] = final_label

            decision = LABEL_MAP[final_label]

            results.append({
                "label": final_label,                     # 'A'/'B'/'C' (from first label token)
                "decision": decision,                     # 'YES'/'NO'/'UNSURE'
                "reason": parsed.get("reason",""),
                "notes":  parsed.get("notes",""),
                "pA": round(batch_pA[i], 6),
                "pB": round(batch_pB[i], 6),
                "pC": round(batch_pC[i], 6),
                "conf_label": final_label,
                "confidence": round(batch_conf[i], 6),    # P(first label token), per §4.4.1 idea
                "context": row.get("context",""),
                "question": row.get("question",""),
                "predicted_answer": row.get("predicted_answer",""),
                "raw": raw,
            })

        torch.cuda.empty_cache()

    return results

# --------------------------
# MAIN
# --------------------------
def main():
    
    import torch, os
    print("Visible GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check drivers / container / nvidia-smi.")

    print(f"[Info] Loading model: {MODEL_PATH}", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_PATH,
        load_in_4bit=True,                  
        dtype=DTYPE,
        device_map={"": "cuda:1"},
        max_seq_length=CTX_LEN,
        attn_implementation="sdpa",
    ) 

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = CTX_LEN

    print(f"[Info] Reading: {INPUT_JSONL}", file=sys.stderr)
    rows = read_jsonl(INPUT_JSONL)[:20]
    print(f"[Info] Judging {len(rows)} items (single-pass, first-label-token confidence: {AGGREGATION})...", file=sys.stderr)

    results = judge_rows_singlepass(model, tokenizer, rows)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"[Done] Saved {len(results)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
