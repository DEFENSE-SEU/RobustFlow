#!/usr/bin/env bash
set -e

BASE_DIR="../../"
AFLOW_DIR="$BASE_DIR/AFlow"
EVAL_DIR="$BASE_DIR/evaluate"
NOISE_DIR="$BASE_DIR/noise_dataset/MATH"

cd "$AFLOW_DIR/workspace/"
tar -czf MATH.tar.gz MATH/
cd "$EVAL_DIR/aflow_scripts/"

run_math_evaluation() {
    local dataset_name=$1
    
    echo "Running evaluation for: $dataset_name"
    
    cd "$AFLOW_DIR/workspace/"
    rm -rf MATH/
    tar -xzf MATH.tar.gz
    
    cd ../data/datasets/
    rm -f math_validate.jsonl
    cp "$NOISE_DIR/${dataset_name}.jsonl" "$AFLOW_DIR/data/datasets/math_validate.jsonl"
    
    cd "$AFLOW_DIR"
    python run.py --dataset MATH --check_convergence False
    mkdir -p "$EVAL_DIR/aflow_scripts/MATH/$dataset_name/"
    cp -a "$AFLOW_DIR/workspace/MATH/workflows/." "$EVAL_DIR/aflow_scripts/MATH/$dataset_name/"
    echo "Completed evaluation for: $dataset_name"
}

datasets=(
    "math_original"
    "math_requirements"
    "math_paraphrasing"
    "math_light_noise"
    "math_moderate_noise"
    "math_heavy_noise"
)

for dataset in "${datasets[@]}"; do
    run_math_evaluation "$dataset"
done

echo "All MATH evaluations completed successfully!"
