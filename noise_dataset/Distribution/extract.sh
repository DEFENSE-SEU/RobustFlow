#!/usr/bin/env bash
set -euo pipefail

datasets=(DROP GSM8K HotpotQA HumanEval MBPP MATH)

variants=(original requirements paraphrasing light_noise moderate_noise heavy_noise)

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
embedding_py="${script_dir}/embedding.py"

out_dir="${script_dir}/embedding"
mkdir -p "${out_dir}"

run_one() {
  local ds="$1"
  local var="$2"
  local lower="${ds,,}"

  local in_file="${script_dir}/../${ds}/${lower}_${var}.jsonl"
  local out_emb="${out_dir}/${lower}_${var}_embeddings.npy"
  local out_q="${out_dir}/${lower}_${var}_questions.jsonl"

  if [[ ! -f "${in_file}" ]]; then
    echo "[SKIP] ${in_file} does not exist, skipping ${ds}:${var}"
    return 0
  fi

  echo "---------------------------------------------"
  echo "[RUN ] dataset=${ds}  variant=${var}"
  echo "       input = ${in_file}"
  echo "       emb   = ${out_emb}"
  echo "       ques  = ${out_q}"
  echo "---------------------------------------------"

  python "${embedding_py}" \
    --input "${in_file}" \
    --dataset "${ds}" \
    --out_embeddings "${out_emb}" \
    --out_questions "${out_q}"
}

for ds in "${datasets[@]}"; do
  for var in "${variants[@]}"; do
    run_one "${ds}" "${var}"
  done
done

echo "✅ All completed. Output written to: ${out_dir}"
