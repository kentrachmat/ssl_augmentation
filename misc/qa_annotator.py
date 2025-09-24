import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# -------------------- CONFIG --------------------
REASONS = ["SPAN_MISSING", "HALLUCINATION", "AMBIGUOUS", "OUT_OF_SCOPE", "DUPLICATE", "OTHER"]
REQUIRED_COLS = [
    "aug_id", "orig_id", "type", "alt_index", "question", "answer",
    "structure_type", "orig_question", "context", "is_valid", "reason", "notes"
]
WORK_DIR = Path("uploaded_store")  # where server copies live
WORK_DIR.mkdir(exist_ok=True)
# ------------------------------------------------

st.set_page_config(page_title="CSV Annotation Tool", layout="wide")

ss = st.session_state
ss.setdefault("df", None)
ss.setdefault("idx", 0)
ss.setdefault("work_path", None)       # server file we save to
ss.setdefault("original_name", None)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            if c == "is_valid":
                df[c] = pd.Series([pd.NA] * len(df), dtype="Int64")   # nullable int
            elif c in ["reason", "notes"]:
                df[c] = pd.Series([""] * len(df), dtype="string")     # force string dtype
            else:
                df[c] = ""
        else:
            # normalize dtype
            if c == "is_valid":
                df[c] = df[c].astype("Int64")
            elif c in ["reason", "notes"]:
                df[c] = df[c].astype("string")
    # order
    return df[[c for c in REQUIRED_COLS]]


def server_path_for(name: str) -> Path:
    # keep original filename, write into WORK_DIR
    base = Path(name).name
    return WORK_DIR / base

def write_csv(path: Path, df: pd.DataFrame, make_backup: bool = True):
    if make_backup:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bkp = path.with_suffix("")  # remove .csv
        backup_path = path.parent / f"{bkp.name}.backup-{ts}.csv"
        df.to_csv(backup_path, index=False, encoding="utf-8")
    df.to_csv(path, index=False, encoding="utf-8")

def clamp_idx(idx: int, n: int) -> int:
    return 0 if n == 0 else max(0, min(idx, n - 1))

def save_and_advance(target_idx: int, advance: str | None, is_valid_str: str, reason: str, notes: str):
    if ss.df is None or ss.work_path is None:
        st.error("No dataset loaded.")
        return

    fresh = ss.df.copy()
    if target_idx >= len(fresh):
        st.error("Row index out of range.")
        return

    # Map UI -> values
    # tri-state select: "—", "1 (valid)", "0 (invalid)"
    if is_valid_str.startswith("1"):
        val_is_valid = 1
    elif is_valid_str.startswith("0"):
        val_is_valid = 0
    else:
        val_is_valid = pd.NA

    val_reason = "" if val_is_valid == 1 else ("" if reason == "—" else reason)
    val_notes = notes

    if val_is_valid == 0 and val_reason == "":
        st.warning("Please choose a reason when marking as 0 (invalid).")
        return

    # Apply
    fresh.at[target_idx, "is_valid"] = val_is_valid
    fresh.at[target_idx, "reason"] = val_reason
    fresh.at[target_idx, "notes"] = val_notes

    # Persist to server path (same filename as uploaded)
    try:
        write_csv(Path(ss.work_path), fresh, make_backup=True)
        st.success(f"Saved to {Path(ss.work_path).resolve()}")
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return

    # Update session + advance
    ss.df = fresh
    if advance == "next":
        ss.idx = clamp_idx(target_idx + 1, len(fresh))
    elif advance == "prev":
        ss.idx = clamp_idx(target_idx - 1, len(fresh))
    else:
        ss.idx = target_idx
    st.rerun()
# ---------------- Upload-only flow ----------------
st.sidebar.header("Upload CSV")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

    df_up = ensure_columns(df_up)
    ss.df = df_up
    ss.original_name = uploaded.name
    ss.work_path = str(server_path_for(uploaded.name))

    # Immediately materialize a working file on server so we can keep overwriting it
    write_csv(Path(ss.work_path), ss.df, make_backup=False)
    st.sidebar.success(f"Loaded: {uploaded.name}")
else:
    if ss.df is None:
        st.info("Upload a CSV to start annotating.")
        st.stop()

df = ss.df

# Header / progress
st.title("Data Augmentation Annotation Tool")
st.markdown(
    'Please refer to this [documentation link](https://www.overleaf.com/project/68cd75f92810718c16e8c347) for more info.'
)
left, right = st.columns([1,1])
with left:
    st.caption(f"Source file: **{ss.original_name}**")
    st.caption(f"Server save path: `{Path(ss.work_path).resolve()}`")
with right:
    completed = df["is_valid"].notna().sum()
    total = len(df)
    st.progress(completed/total if total else 0.0, text=f"{completed}/{total} completed")

# Navigation (no jump box)
nav_prev, nav_next = st.columns([1,1])
with nav_prev:
    if st.button("Previous"):
        ss.idx = clamp_idx(ss.idx - 1, len(df))
with nav_next:
    if st.button("Next"):
        ss.idx = clamp_idx(ss.idx + 1, len(df))

idx = ss.idx
if len(df) == 0:
    st.warning("CSV is empty.")
    st.stop()

row = df.iloc[idx].copy()
# ===================== Display (reordered) =====================
st.subheader(f"Row {idx+1} / {len(df)}")

top1, top2, top3 = st.columns([1,1,1])
with top1:
    st.markdown(f"**aug_id**: `{row.get('aug_id', '')}`")
with top2:
    st.markdown(f"**type**: `{row.get('type', '')}`")
with top3:
    st.markdown(f"**structure_type**:")
    st.code(str(row.get("structure_type", "")), language=None)

# "type context" (use 'context' but label it clearly)
st.markdown("**Type Context**")
with st.expander("Show / Hide context", expanded=False):
    st.write(str(row.get("context", "")))

# Original Q/A (if orig_answer doesn't exist, show blank)
st.markdown("**Original Question**")
st.write(str(row.get("orig_question", "")))

st.markdown("**Original Answer**")
st.write(str(row.get("orig_answer", "")))

st.markdown("---")

# Current Q/A
st.markdown("### Current Item")
ca1, ca2 = st.columns(2)
with ca1:
    st.markdown("**Question**")
    st.write(str(row.get("question", "")))
with ca2:
    st.markdown("**Answer**")
    st.write(str(row.get("answer", "")))

st.markdown("---")

# ===================== Annotation form (two buttons) =====================
with st.form("annotation_form", clear_on_submit=False):
    c1, c2 = st.columns([1, 2])

    # current value -> ui label
    current_valid = row["is_valid"]
    if pd.isna(current_valid):
        default_valid_label = "—"
    elif int(current_valid) == 1:
        default_valid_label = "1 (valid)"
    else:
        default_valid_label = "0 (invalid)"

    valid_label = c1.selectbox(
        "Is valid?",
        options=["—", "1 (valid)", "0 (invalid)"],
        index=["—", "1 (valid)", "0 (invalid)"].index(default_valid_label),
        help="Pick 1 (valid) or 0 (invalid)."
    )
    
    if valid_label == "—":
        st.caption(":red[Please select 1 or 0 to continue.]")


    reason_opts = ["—"] + REASONS
    default_reason = "—" if pd.isna(row["reason"]) or str(row["reason"]).strip()=="" else str(row["reason"])
    if default_reason not in reason_opts:
        reason_opts.append(default_reason)  # preserve unexpected existing value
    reason_sel = c1.selectbox(
        "Reason (if invalid)",
        options=reason_opts,
        index=reason_opts.index(default_reason) if default_reason in reason_opts else 0,
        help="Required if you mark the item invalid."
    )

    notes_text = c2.text_area(
        "Notes (free text)",
        value="" if pd.isna(row["notes"]) else str(row["notes"]),
        height=120
    )

    b_prev, b_next = st.columns([1,1])
    save_prev = b_prev.form_submit_button("⏮ Save & Previous")
    save_next = b_next.form_submit_button("✅ Save & Next")

# Save handler (unchanged except no 'save_only')
if save_prev or save_next:
    # Require a selection for "Is valid?"
    if valid_label == "—":
        st.error("Please select **Is valid?** (1 or 0) before saving.")
        # Do NOT advance or save
        st.stop()  # or `return` if inside a function
    move = "next" if save_next else "prev"
    save_and_advance(idx, move, valid_label, reason_sel, notes_text)


# Quick nav (keep if you still want buttons outside the form)
colA, colC = st.columns([1,1])
with colA:
    st.button("⏮ Previous", on_click=lambda: ss.update(idx=clamp_idx(ss.idx - 1, len(ss.df))))
with colC:
    st.button("Next ⏭", on_click=lambda: ss.update(idx=clamp_idx(ss.idx + 1, len(ss.df))))
