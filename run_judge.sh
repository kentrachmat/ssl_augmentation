#!/bin/bash
set -e

# Paths
SQUAD_INPUT="final/ds_squadv2_aug.csv"
TECHQA_INPUT="final/ds_techqa_aug.csv"

# Run for SQuADv2
for i in 1
do
    echo "[Info] Running judge for annotator chatgpt$i on SQuADv2 ..."
    python llm_as_judge.py \
        --input "$SQUAD_INPUT" \
        --name "chatgpt$i" \
        --dataset "squadv2"
done

# Run for TechQA
for i in 1
do
    echo "[Info] Running judge for annotator chatgpt$i on TechQA ..."
    python llm_as_judge.py \
        --input "$TECHQA_INPUT" \
        --name "chatgpt$i" \
        --dataset "techqa"
done

echo "[Done] All 2 runs completed."