from __future__ import annotations
import re, json, math, string, time
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ---------------- Normalization & QA metrics ----------------
_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def _remove_articles(s: str) -> str: return " ".join(w for w in s.split() if w not in _ARTICLES)
def _white_space_fix(s: str) -> str: return " ".join(s.split())
def _remove_punc(s: str) -> str: return s.translate(_PUNCT_TABLE)
def _lower(s: str) -> str: return s.lower()

def normalize_answer(s: str) -> str:
    if s is None: return ""
    return _white_space_fix(_remove_articles(_remove_punc(_lower(s))))

_NO_ANSWER_CANON = {"", "no answer", "unanswerable", "n/a", "none", "null", "na", "not available"}
def canon_no_answer(s: str) -> str:
    t = normalize_answer(s)
    return "" if t in _NO_ANSWER_CANON else t

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = canon_no_answer(prediction).split()
    gold_tokens = canon_no_answer(ground_truth).split()
    if len(gold_tokens) == 0 and len(pred_tokens) == 0: return 1.0
    if len(gold_tokens) == 0 or len(pred_tokens) == 0: return 0.0
    common = {}
    for t in gold_tokens: common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in pred_tokens:
        if common.get(t, 0) > 0:
            num_same += 1; common[t] -= 1
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(pred: str, gold: str) -> float:
    return 1.0 if canon_no_answer(pred) == canon_no_answer(gold) else 0.0

# ---------------- Embedding similarity (single gold) ----------------
EMBED_MODEL_NAME = "/home/brachmat/phd/models/models--qwen3-embedding-0.6B"

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def sem_sim_single(pred: str, gold: str, embedder) -> float:
    if canon_no_answer(pred) == "" and canon_no_answer(gold) == "":
        return 1.0
    vecs = embedder.encode([pred, gold], normalize_embeddings=False, convert_to_numpy=True)
    return cosine_sim(vecs[0], vecs[1])

# ---------------- Filename parsing (your schema) ----------------
FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<model>[^_]+)_(?P<method>[^_]+)_(?P<sample>N\d+)_(?P<gpus>gpus\d+)$",
    re.IGNORECASE,
)

def parse_run_filename(stem: str) -> Dict[str, str]:
    m = FILENAME_RE.match(stem)
    if not m:
        parts = stem.split("_")
        return {
            "timestamp": parts[0] + ("_" + parts[1] if len(parts) > 1 and parts[1].isdigit() else ""),
            "model": next((p for p in parts if p.lower() in {"llama","qwen","mistral","gemma","phi"}), "unknown"),
            "method": next((p for p in parts if p.lower() in {"baseline","semantic","syntactic","lexical"}), "unknown"),
            "sample": next((p for p in parts if p.startswith("N")), "N0"),
            "gpus": next((p for p in parts if p.startswith("gpus")), "gpus1"),
        }
    d = m.groupdict()
    return {
        "timestamp": f"{d['date']}_{d['time']}",
        "model": d["model"].lower(),
        "method": d["method"].lower(),
        "sample": d["sample"],
        "gpus": d["gpus"],
    }

# ---------------- Core with PROGRESS (single-gold ground truth) ----------------
def evaluate_single_dataset_root(
    root_dir: str | Path,
    ground_truth_csv: str | Path,
    dataset_name: Optional[str] = None,
    embed_model_name: str = EMBED_MODEL_NAME,
    save_csv_path: Optional[str | Path] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    ROOT contains runs for a single dataset (either SQuAD or PubMed).
    Shows file-level and row-level progress bars.
    Returns tidy DF with:
      dataset, file, timestamp, model, method, sample, gpus, n, EM, F1, SemSim
    """
    t_start = time.time()
    root_dir = Path(root_dir)

    # Infer dataset if not provided
    if dataset_name is None:
        tokens = {p.lower() for p in root_dir.parts}
        if "squad" in tokens: dataset_name = "squad"
        elif "pubmed" in tokens: dataset_name = "pubmed"
        else:
            kids = {p.name.lower() for p in root_dir.iterdir() if p.is_dir()}
            if "squad" in kids: dataset_name = "squad"
            elif "pubmed" in kids: dataset_name = "pubmed"
            else: dataset_name = "unknown"

    gt = pd.read_csv(ground_truth_csv)
    # Expect single gold in 'answer'
    if "id" not in gt.columns or "answer" not in gt.columns:
        raise ValueError("Ground-truth CSV must have columns: id, answer (single string).")
    gt = gt[["id", "answer"]].copy()
    # Normalize NaNs to empty string to avoid crashes
    gt["answer"] = gt["answer"].astype(str).replace({"nan": ""})

    embedder = SentenceTransformer(embed_model_name)
    rows = []

    all_csvs = list(root_dir.rglob("*.csv"))
    file_bar = tqdm(all_csvs, desc=f"[{dataset_name}] files", disable=quiet)

    skipped = 0
    total_pairs = 0

    for f in file_bar:
        df = pd.read_csv(f)
        if "id" not in df.columns or "generated_answer" not in df.columns:
            skipped += 1
            file_bar.write(f"[SKIP schema] {f.name} (needs columns [id, generated_answer])")
            continue

        merged = gt.merge(df[["id", "generated_answer"]], on="id", how="inner", validate="one_to_one")
        if merged.empty:
            skipped += 1
            file_bar.write(f"[SKIP empty] {f.name} (no id overlap)")
            continue

        ems, f1s, sims = [], [], []
        row_bar = tqdm(range(len(merged)), desc=f"→ {f.name}", leave=False, disable=quiet)
        for i in row_bar:
            pred = "" if pd.isna(merged.iloc[i]["generated_answer"]) else str(merged.iloc[i]["generated_answer"])
            gold = "" if pd.isna(merged.iloc[i]["answer"]) else str(merged.iloc[i]["answer"])
            em = exact_match_score(pred, gold)
            f1 = f1_score(pred, gold)
            sim = sem_sim_single(pred, gold, embedder)
            ems.append(em); f1s.append(f1); sims.append(sim)

        meta = parse_run_filename(f.stem)
        n_pairs = len(merged)
        total_pairs += n_pairs
        rows.append({
            "dataset": dataset_name,
            "file": f.name,
            "timestamp": meta["timestamp"],
            "model": meta["model"],
            "method": meta["method"],
            "sample": meta["sample"],
            "gpus": meta["gpus"],
            "n": n_pairs,
            "EM": float(np.mean(ems)),
            "F1": float(np.mean(f1s)),
            "SemSim": float(np.mean(sims)),
        })

        file_bar.set_postfix(last_file=f.name, n=n_pairs, em=f"{rows[-1]['EM']:.3f}", f1=f"{rows[-1]['F1']:.3f}")

    if not rows:
        raise RuntimeError("No valid runs found (check ROOT, filenames, and CSV schemas).")

    out = pd.DataFrame(rows).sort_values(
        ["model", "method", "timestamp", "EM", "F1", "SemSim"],
        ascending=[True, True, True, False, False, False],
    )
    if save_csv_path:
        out.to_csv(save_csv_path, index=False)
        if not quiet:
            print(f"[Saved] {save_csv_path}")

    if not quiet:
        dur = time.time() - t_start
        print(f"[{dataset_name}] processed {len(rows)} files, {total_pairs} Q/A pairs, skipped {skipped} files in {dur:.1f}s.")

    return out


if __name__ == "__main__":
    QUIET = False

    ROOT_PUBMED = "/home/brachmat/phd/ssl_augmentation/inference_outputs_unsloth/pubmed"
    GT_PUBMED   = "/home/brachmat/phd/datasets/pubmed_final/test.csv"
    df_pubmed = evaluate_single_dataset_root(
        ROOT_PUBMED, GT_PUBMED, dataset_name="pubmed", save_csv_path="pubmed_runs_metrics.csv", quiet=QUIET
    )

    ROOT_SQUAD = "/home/brachmat/phd/ssl_augmentation/inference_outputs_unsloth/squad"
    GT_SQUAD   = "/home/brachmat/phd/datasets/squad_v2_final/test.csv"
    df_squad = evaluate_single_dataset_root(
        ROOT_SQUAD, GT_SQUAD, dataset_name="squad", save_csv_path="squad_runs_metrics.csv", quiet=QUIET
    )

    df_all = pd.concat([df_squad, df_pubmed], ignore_index=True)
    df_all.to_csv("all_runs_metrics.csv", index=False)
    print("\nTop 20 rows:")
    print(df_all.head(20).to_string(index=False))
    print(f"\n[ALL] saved all_runs_metrics.csv with {len(df_all)} rows.")
