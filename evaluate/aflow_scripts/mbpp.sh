#!/usr/bin/env bash
set -e

BASE_DIR="../../"
AFLOW_DIR="$BASE_DIR/AFlow"
EVAL_DIR="$BASE_DIR/evaluate"
NOISE_DIR="$BASE_DIR/noise_dataset/MBPP"

cd "$AFLOW_DIR/workspace/"
tar -czf MBPP.tar.gz MBPP/
cd "$EVAL_DIR/aflow_scripts/"

run_mbpp_evaluation() {
    local dataset_name=$1
    
    echo "Running evaluation for: $dataset_name"
    
    cd "$AFLOW_DIR/workspace/"
    rm -rf MBPP/
    tar -xzf MBPP.tar.gz
    
    cd ../data/datasets/
    rm -f mbpp_validate.jsonl
    cp "$NOISE_DIR/${dataset_name}.jsonl" "$AFLOW_DIR/data/datasets/mbpp_validate.jsonl"
    
    cd "$AFLOW_DIR"
    python run.py --dataset MBPP --check_convergence False
    mkdir -p "$EVAL_DIR/aflow_scripts/MBPP/$dataset_name/"
    cp -a "$AFLOW_DIR/workspace/MBPP/workflows/." "$EVAL_DIR/aflow_scripts/MBPP/$dataset_name/"
    echo "Completed evaluation for: $dataset_name"
}

datasets=(
    "mbpp_original"
    "mbpp_requirements"
    "mbpp_paraphrasing"
    "mbpp_light_noise"
    "mbpp_moderate_noise"
    "mbpp_heavy_noise"
)

for dataset in "${datasets[@]}"; do
    run_mbpp_evaluation "$dataset"
done

echo "All MBPP evaluations completed successfully!"
