import os
import json

TASK_DIRS = ["DROP", "GSM8K", "HotpotQA", "HumanEval", "MATH", "MBPP"]

def select_best_round(data, path):
    """Select the best entry based on score, avg_cost, total_cost, and round"""
    if not data:
        print(f"[WARN] {path} is empty file")
        return None

    best = sorted(data, key=lambda x: (
        x.get("score", 0),
        x.get("avg_cost", 0),
        x.get("total_cost", 0),
        x.get("round", 0)
    ), reverse=True)[0]

    return best


def find_and_process_results(root_path):
    """Traverse all task directories under the specified path and find and analyze results.json files"""
    output_file = os.path.join(root_path, "best.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for task in sorted(TASK_DIRS):
            task_path = os.path.join(root_path, task)

            info_msg = f"\n[INFO] Entering task directory: {task_path}"
            print(info_msg)
            f.write(info_msg + "\n")
            
            for subfolder in sorted(os.listdir(task_path)):
                subfolder_path = os.path.join(task_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                result_file = os.path.join(subfolder_path, "results.json")
                if os.path.isfile(result_file):
                    try:
                        with open(result_file, "r", encoding="utf-8") as json_file:
                            data = json.load(json_file)
                        best = select_best_round(data, result_file)
                        if best:
                            best_msg = f"[BEST] {result_file}/round {best['round']} (score={best['score']:.4f})"
                            print(best_msg)
                            f.write(best_msg + "\n")
                        else:
                            empty_msg = f"[EMPTY] {result_file} has no valid records"
                            print(empty_msg)
                            f.write(empty_msg + "\n")
                    except Exception as e:
                        error_msg = f"[ERROR] Failed to read: {result_file}, reason: {e}"
                        print(error_msg)
                        f.write(error_msg + "\n")
                else:
                    miss_msg = f"[MISS] Not found: {result_file}"
                    print(miss_msg)
                    f.write(miss_msg + "\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    find_and_process_results(current_dir)
