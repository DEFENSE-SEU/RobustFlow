#!/usr/bin/env bash

set -u
mkdir -p logs

scripts=(
  "aflow_scripts/drop.sh"
  "aflow_scripts/gsm8k.sh"
  "aflow_scripts/hotpotqa.sh"
  "aflow_scripts/humaneval.sh"
  "aflow_scripts/math.sh"
  "aflow_scripts/mbpp.sh"
)

declare -A pids=()
declare -A status=()

for s in "${scripts[@]}"; do
  if [[ ! -f "$s" ]]; then
    echo "[WARN] $s does not exist, skipping"
    status["$s"]=127
    continue
  fi
  
  base="${s%.sh}"
  log="logs/${base}.log"
  echo "[RUN] $s  -> $log"
  
  stdbuf -oL -eL bash "$s" >"$log" 2>&1 &
  
  pid=$!
  pids["$pid"]="$s"
done

trap 'echo; echo "[INT] stopping all..."; for pid in "${!pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; wait; exit 130' INT TERM

for pid in "${!pids[@]}"; do
  s="${pids[$pid]}"
  
  if wait "$pid"; then
    status["$s"]=0
    echo "[OK ] $s"
  else
    rc=$?
    status["$s"]=$rc
    echo "[FAIL] $s (exit $rc)"
  fi
done

echo "================ SUMMARY ================"
ret=0

for s in "${scripts[@]}"; do
  rc="${status[$s]:-NA}"
  printf "%-20s -> %s\n" "$s" "$rc"
  
  [[ "$rc" != "0" ]] && ret=1
done

exit "$ret"