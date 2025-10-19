import json
import sys
from pathlib import Path
import textwrap


def iter_jsonl(file_path: str):
    """
    逐行读取 JSONL 文件并解析为 Python 对象（dict/list/...）
    - 跳过空行、以 # 或 // 开头的注释行
    - 解析失败时打印警告但不中断
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {lineno}: {e}", file=sys.stderr)

def json_to_question(data: dict) -> str:
    """
    将一个 JSON 对象拼接成一整个字符串形式的问题
    """
    parts = []
    
    parts.append(textwrap.dedent(f"""
    You will complete a programming problem. Please implement the function `{data['entry_point']}` in python.
    """).strip())
    
    parts.append("Problem:\n" + data["prompt"].strip() + "\n")
    parts.append("TESTS:\n" + data["test"].strip() + "\n")
    return "\n\n".join(parts)

if __name__ == "__main__":
    # original_path = "../../ScoreFlow/data/humaneval_original.jsonl"
    # requirement_path = "../../ScoreFlow/data/humaneval_requirements.jsonl"
    # paraphrasing_path = "../../ScoreFlow/data/humaneval_paraphrasing.jsonl"
    light_noise_path = "../../ScoreFlow/data/mbpp_light_noise.jsonl"
    moderate_noise_path = "../../ScoreFlow/data/mbpp_moderate_noise.jsonl"
    # heavy_noise_path = "../../ScoreFlow/data/humaneval_heavy_noise.jsonl"
    all_questions = []
    # for obj in iter_jsonl(original_path):
    #     all_questions.append({"question": json_to_question(obj)})
    # for obj in iter_jsonl(requirement_path):
    #     all_questions.append({"question": json_to_question(obj)})
    # for obj in iter_jsonl(paraphrasing_path):
    #     all_questions.append({"question": json_to_question(obj)})
    for obj in iter_jsonl(light_noise_path):
        all_questions.append({"question": json_to_question(obj)})
    for obj in iter_jsonl(moderate_noise_path):
        all_questions.append({"question": json_to_question(obj)})
    # for obj in iter_jsonl(heavy_noise_path):
    #     all_questions.append({"question": json_to_question(obj)})
    light_noise_path = "../../ScoreFlow/data/humaneval_light_noise.jsonl"
    moderate_noise_path = "../../ScoreFlow/data/humaneval_moderate_noise.jsonl"
    for obj in iter_jsonl(light_noise_path):
        all_questions.append({"question": json_to_question(obj)})
    for obj in iter_jsonl(moderate_noise_path):
        all_questions.append({"question": json_to_question(obj)})
    with open("all_questions.jsonl", "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False))  # 内部换行会被转义为 \n
            f.write("\n")