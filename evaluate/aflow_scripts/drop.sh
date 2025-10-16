#!/usr/bin/env bash
set -e

BASE_DIR="../../"
AFLOW_DIR="$BASE_DIR/AFlow"
EVAL_DIR="$BASE_DIR/evaluate"
NOISE_DIR="$BASE_DIR/noise_dataset/DROP"

cd "$AFLOW_DIR/workspace/"
tar -czf DROP.tar.gz DROP/
cd "$EVAL_DIR/aflow_scripts/"

run_drop_evaluation() {
    local dataset_name=$1
    
    echo "Running evaluation for: $dataset_name"
    
    cd "$AFLOW_DIR/workspace/"
    rm -rf DROP/
    tar -xzf DROP.tar.gz
    
    cd ../data/datasets/
    rm -f drop_validate.jsonl
    cp "$NOISE_DIR/${dataset_name}.jsonl" "$AFLOW_DIR/data/datasets/drop_validate.jsonl"
    
    cd "$AFLOW_DIR"
    python run.py --dataset DROP --check_convergence False
    mkdir -p "$EVAL_DIR/aflow_scripts/DROP/$dataset_name/"
    cp -a "$AFLOW_DIR/workspace/DROP/workflows/." "$EVAL_DIR/aflow_scripts/DROP/$dataset_name/"
    echo "Completed evaluation for: $dataset_name"
}

datasets=(
    "drop_original"
    "drop_requirements" 
    "drop_paraphrasing"
    "drop_light_noise"
    "drop_moderate_noise"
    "drop_heavy_noise"
)

for dataset in "${datasets[@]}"; do
    run_drop_evaluation "$dataset"
done

echo "All DROP evaluations completed successfully!"