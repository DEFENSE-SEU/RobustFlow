import argparse
import json
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from typing import Tuple, List, Optional
from sentence_transformers import SentenceTransformer

DROP_PATTERN = re.compile(r"Question\s*:\s*(.*?)\s*\n\s*Answer\s*:", flags=re.DOTALL)
GSM8K_PATTERN = re.compile(r"^(.*)$", flags=re.DOTALL)
HotpotQA_PATTERN = re.compile(r"^(.*)$", flags=re.DOTALL)
HumanEval_PATTERN = re.compile(r"^(.*)$", flags=re.DOTALL)
MATH_PATTERN = re.compile(r"^(.*)$", flags=re.DOTALL)
MBPP_PATTERN = re.compile(r'^(.*?)\s*def', flags=re.DOTALL)

DATASET_FIELD_PATTERN = {
    "DROP":      ("context",  DROP_PATTERN),
    "GSM8K":     ("question", GSM8K_PATTERN),
    "HOTPOTQA":  ("question", HotpotQA_PATTERN),
    "HUMANEVAL": ("prompt",   HumanEval_PATTERN),
    "MBPP":      ("prompt",   MBPP_PATTERN),
    "MATH":      ("problem",  MATH_PATTERN),
}

def extract_question(context: str, pattern: re.Pattern) -> Optional[str]:
    if not isinstance(context, str):
        return None
    m = pattern.search(context)
    if not m:
        return None
    return (m.group(1) if pattern.groups >= 1 else m.group(0)).strip()

def read_questions_from_jsonl(jsonl_path: Path, dataset: str) -> Tuple[List[str], List[int]]:
    key = dataset.strip().upper()
    if key not in DATASET_FIELD_PATTERN:
        raise ValueError(f"Unsupported dataset: {dataset}. "
                         f"Supported: {', '.join(DATASET_FIELD_PATTERN.keys())}")

    field_name, pattern = DATASET_FIELD_PATTERN[key]

    questions, ids = [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            context = obj.get(field_name, "")
            q = extract_question(context, pattern)
            if q:
                questions.append(q)
                ids.append(len(questions))
    return questions, ids

def main():
    parser = argparse.ArgumentParser(description="Extract questions and compute embeddings.")
    parser.add_argument("--input", type=str, default="../Noise_Dataset/DROP/drop_original.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--dataset", type=str.upper, default="DROP", choices=["DROP", "GSM8K", "HOTPOTQA", "HUMANEVAL", "MBPP", "MATH"],
        help="Dataset name (determines field extraction and regex): drop/gsm8k/hotpotqa/humaneval/mbpp/math")
    parser.add_argument("--out_embeddings", type=str, default="drop_embeddings.npy",
                        help="Output embeddings matrix .npy file path")
    parser.add_argument("--out_questions", type=str, default="drop_questions.jsonl",
                        help="Output extracted questions .jsonl file (contains id and question)")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--normalize", action="store_true",
                        help="Whether to L2 normalize embeddings (recommended for similarity search)")
    args = parser.parse_args()

    input_path = Path(args.input)
    assert input_path.exists(), f"Input file does not exist: {input_path}"

    questions, ids = read_questions_from_jsonl(input_path, dataset=args.dataset)
    print(f"[{args.dataset}] Reading completed: extracted {len(questions)} questions.")

    if not questions:
        print("No questions extracted. Please check JSONL format and regex pattern.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to(device)
    print(f"Model loaded to device: {device}")

    embeddings = model.encode(
        questions,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=args.normalize
    )
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(args.out_embeddings, embeddings)
    print(f"Embeddings saved to: {args.out_embeddings}")

    with open(args.out_questions, "w", encoding="utf-8") as wf:
        for i, q in zip(ids, questions):
            wf.write(json.dumps({"id": i, "question": q}, ensure_ascii=False) + "\n")
    print(f"Questions saved to: {args.out_questions}")

if __name__ == "__main__":
    main()
