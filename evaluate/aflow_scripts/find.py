# find.py

import os
import json

# 顶层任务目录
TASK_DIRS = ["DROP", "GSM8K", "HotpotQA", "HumanEval", "MATH", "MBPP"]

def select_best_round(data, path):
    """根据 score、avg_cost、total_cost、round 选择最优的条目"""
    if not data:
        print(f"[WARN] {path} 是空文件")
        return None

    # 多重排序：score DESC, avg_cost DESC, total_cost DESC, round DESC
    best = sorted(data, key=lambda x: (
        x.get("score", 0),
        x.get("avg_cost", 0),
        x.get("total_cost", 0),
        x.get("round", 0)
    ), reverse=True)[0]

    return best


def find_and_process_results(root_path):
    for task in sorted(TASK_DIRS):
        task_path = os.path.join(root_path, task)
        if not os.path.isdir(task_path):
            print(f"[WARN] 跳过不存在的目录: {task_path}")
            continue

        print(f"\n[INFO] 进入任务目录: {task_path}")
        for subfolder in sorted(os.listdir(task_path)):
            subfolder_path = os.path.join(task_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            result_file = os.path.join(subfolder_path, "results.json")
            if os.path.isfile(result_file):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    best = select_best_round(data, result_file)
                    if best:
                        print(f"[BEST] {result_file}/round {best['round']} "
                              f"(score={best['score']:.4f})")
                    else:
                        print(f"[EMPTY] {result_file} 无有效记录")
                except Exception as e:
                    print(f"[ERROR] 读取失败: {result_file}，原因: {e}")
            else:
                print(f"[MISS] 未找到: {result_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    find_and_process_results(current_dir)
