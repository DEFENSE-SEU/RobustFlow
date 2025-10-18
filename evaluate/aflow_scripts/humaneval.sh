#!/usr/bin/env bash
set -e

BASE_DIR="../../"
AFLOW_DIR="$BASE_DIR/AFlow"
EVAL_DIR="$BASE_DIR/evaluate"
NOISE_DIR="$BASE_DIR/noise_dataset/HumanEval"

cd "$AFLOW_DIR/workspace/"
tar -czf HumanEval.tar.gz HumanEval/
cd "$EVAL_DIR/aflow_scripts/"

run_humaneval_evaluation() {
    local dataset_name=$1
    
    echo "Running evaluation for: $dataset_name"
    
    cd "$AFLOW_DIR/workspace/"
    rm -rf HumanEval/
    tar -xzf HumanEval.tar.gz
    
    cd ../data/datasets/
    rm -f humaneval_validate.jsonl
    cp "$NOISE_DIR/${dataset_name}.jsonl" "$AFLOW_DIR/data/datasets/humaneval_validate.jsonl"
    
    cd "$AFLOW_DIR"
    python run.py --dataset HumanEval --check_convergence False
    mkdir -p "$EVAL_DIR/aflow_scripts/HumanEval/$dataset_name/"
    cp -a "$AFLOW_DIR/workspace/HumanEval/workflows/." "$EVAL_DIR/aflow_scripts/HumanEval/$dataset_name/"
    echo "Completed evaluation for: $dataset_name"
}

datasets=(
    "humaneval_original"
    "humaneval_requirements"
    "humaneval_paraphrasing"
    "humaneval_light_noise"
    "humaneval_moderate_noise"
    "humaneval_heavy_noise"
)

for dataset in "${datasets[@]}"; do
    run_humaneval_evaluation "$dataset"
done

echo "All HumanEval evaluations completed successfully!"
