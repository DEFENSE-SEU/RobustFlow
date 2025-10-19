#!/usr/bin/env bash  # 指定使用bash解释器执行此脚本
set -euo pipefail  # 设置严格的错误处理：-e遇到错误立即退出，-u未定义变量报错，-o pipefail管道中任一命令失败则整个管道失败

FILE="/data/dishimin/RobustFlow/Flow/data/all_questions.jsonl"  # 定义输入文件路径，包含所有问题的JSONL格式数据
OUT="/data/dishimin/RobustFlow/Flow/initflow.json"  # 定义输出文件路径，Flow模型处理后的结果文件
RESULTS_DIR="/data/dishimin/RobustFlow/Flow/results"  # 定义结果目录路径，用于存储所有处理结果
mkdir -p "$RESULTS_DIR"  # 创建结果目录，-p参数确保父目录不存在时一并创建，目录已存在时不报错

num=1  # 初始化计数器变量，用于跟踪当前处理的条目编号

jq -rc '.question | @json' "$FILE" | while IFS= read -r qjson; do  # 使用jq工具提取JSONL文件中每个条目的question字段，通过管道逐行读取
    echo "[`date '+%F %T'`] start item #$num"  # 打印带时间戳的开始处理信息，显示当前处理的条目编号

    # 超过 30 分钟自动停止 main.py，但不让整个脚本退出
    # 说明：timeout 超时返回码为 124；此处统一打印一下并继续后续 cp
    if ! timeout 10m python "/data/dishimin/RobustFlow/Flow/main.py" <<<"$qjson"; then  # 使用timeout命令限制main.py执行时间为10分钟，通过here string传递问题JSON数据，如果命令失败则执行then块
        rc=$?  # 获取上一个命令的退出状态码
        if [[ $rc -eq 124 ]]; then  # 检查退出码是否为124（timeout超时退出码）
            echo "[`date '+%F %T'`] item #$num TIMEOUT (30m), continue to copy." >&2  # 如果是超时，打印超时信息到标准错误输出，注意注释说的是30分钟但实际设置的是10分钟
        else  # 如果不是超时退出码
            echo "[`date '+%F %T'`] item #$num python exited with code $rc, continue to copy." >&2  # 打印Python程序异常退出的信息，包含具体的退出码
        fi  # 结束if语句块
    fi  # 结束if语句块，无论Python程序是否成功执行，都会继续执行后续的复制操作

    cp -f "$OUT" "$RESULTS_DIR/initflow${num}.json"  # 强制复制输出文件到结果目录，使用条目编号作为文件名后缀，-f参数强制覆盖已存在文件
    echo "[`date '+%F %T'`] wrote $RESULTS_DIR/initflow${num}.json"  # 打印带时间戳的文件写入确认信息

    num=$((num+1))  # 将计数器加1，准备处理下一个条目
done  # 结束while循环，当所有问题处理完毕时退出
