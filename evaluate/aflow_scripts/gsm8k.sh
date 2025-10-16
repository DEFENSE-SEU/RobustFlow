#!/usr/bin/env bash
set -e

BASE_DIR="../../"
AFLOW_DIR="$BASE_DIR/AFlow"
EVAL_DIR="$BASE_DIR/evaluate"
NOISE_DIR="$BASE_DIR/noise_dataset/GSM8K"

cd "$AFLOW_DIR/workspace/"
tar -czf GSM8K.tar.gz GSM8K/
cd "$EVAL_DIR/aflow_scripts/"

run_gsm8k_evaluation() {
    local dataset_name=$1
    
    echo "Running evaluation for: $dataset_name"
    
    cd "$AFLOW_DIR/workspace/"
    rm -rf GSM8K/
    tar -xzf GSM8K.tar.gz
    
    cd ../data/datasets/
    rm -f gsm8k_validate.jsonl
    cp "$NOISE_DIR/${dataset_name}.jsonl" "$AFLOW_DIR/data/datasets/gsm8k_validate.jsonl"
    
    cd "$AFLOW_DIR"
    python run.py --dataset GSM8K --check_convergence False
    mkdir -p "$EVAL_DIR/aflow_scripts/GSM8K/$dataset_name/"
    cp -a "$AFLOW_DIR/workspace/GSM8K/workflows/." "$EVAL_DIR/aflow_scripts/GSM8K/$dataset_name/"
    echo "Completed evaluation for: $dataset_name"
}

datasets=(
    "gsm8k_original"
    "gsm8k_requirements"
    "gsm8k_paraphrasing"
    "gsm8k_light_noise"
    "gsm8k_moderate_noise"
    "gsm8k_heavy_noise"
)

for dataset in "${datasets[@]}"; do
    run_gsm8k_evaluation "$dataset"
done

echo "All GSM8K evaluations completed successfully!"