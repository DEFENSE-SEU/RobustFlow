#!/usr/bin/env bash
set -euo pipefail

FILE="/data/dishimin/RobustFlow/Flow/data/all_questions.jsonl"
OUT="/data/dishimin/RobustFlow/Flow/initflow.json"
RESULTS_DIR="/data/dishimin/RobustFlow/Flow/results"
mkdir -p "$RESULTS_DIR"

num=1

jq -rc '.question | @json' "$FILE" | while IFS= read -r qjson; do
    echo "[`date '+%F %T'`] start item #$num"

    # 超过 30 分钟自动停止 main.py，但不让整个脚本退出
    # 说明：timeout 超时返回码为 124；此处统一打印一下并继续后续 cp
    if ! timeout 10m python "/data/dishimin/RobustFlow/Flow/main.py" <<<"$qjson"; then
        rc=$?
        if [[ $rc -eq 124 ]]; then
            echo "[`date '+%F %T'`] item #$num TIMEOUT (30m), continue to copy." >&2
        else
            echo "[`date '+%F %T'`] item #$num python exited with code $rc, continue to copy." >&2
        fi
    fi

    cp -f "$OUT" "$RESULTS_DIR/initflow${num}.json"
    echo "[`date '+%F %T'`] wrote $RESULTS_DIR/initflow${num}.json"

    num=$((num+1))
done
