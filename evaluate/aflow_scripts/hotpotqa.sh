#!/usr/bin/env bash
set -e

BASE_DIR="../../"
AFLOW_DIR="$BASE_DIR/AFlow"
EVAL_DIR="$BASE_DIR/evaluate"
NOISE_DIR="$BASE_DIR/noise_dataset/HotpotQA"

cd "$AFLOW_DIR/workspace/"
tar -czf HotpotQA.tar.gz HotpotQA/
cd "$EVAL_DIR/aflow_scripts/"

run_hotpotqa_evaluation() {
    local dataset_name=$1
    
    echo "Running evaluation for: $dataset_name"
    
    cd "$AFLOW_DIR/workspace/"
    rm -rf HotpotQA/
    tar -xzf HotpotQA.tar.gz
        
    cd ../data/datasets/
    rm -f hotpotqa_validate.jsonl
    cp "$NOISE_DIR/${dataset_name}.jsonl" "$AFLOW_DIR/data/datasets/hotpotqa_validate.jsonl"
    
    cd "$AFLOW_DIR"
    python run.py --dataset HotpotQA --check_convergence False
    mkdir -p "$EVAL_DIR/aflow_scripts/HotpotQA/$dataset_name/"
    cp -a "$AFLOW_DIR/workspace/HotpotQA/workflows/." "$EVAL_DIR/aflow_scripts/HotpotQA/$dataset_name/"
    echo "Completed evaluation for: $dataset_name"
}

datasets=(
    "hotpotqa_original"
    "hotpotqa_requirements"
    "hotpotqa_paraphrasing"
    "hotpotqa_light_noise"
    "hotpotqa_moderate_noise"
    "hotpotqa_heavy_noise"
)

for dataset in "${datasets[@]}"; do
    run_hotpotqa_evaluation "$dataset"
done

echo "All HotpotQA evaluations completed successfully!"
