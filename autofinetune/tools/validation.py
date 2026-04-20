import json
from pathlib import Path
from loguru import logger
from autofinetune.tools.data_cleaning import _load_jsonl


def get_dataset_stats(dataset_path: str) -> dict:
    """
    Already defined in data_cleaning.py.
    Re-exported here for the tool registry.
    """
    from autofinetune.tools.data_cleaning import get_dataset_stats as _get_stats
    return _get_stats(dataset_path)


def check_leakage(
    benchmark_path: str,
    train_dataset_path: str,
    threshold: float = 0.8,
) -> dict:
    """
    Checks if benchmark examples are too similar to training data.
    Uses character n-gram overlap as a fast proxy.
    """
    benchmark = _load_jsonl(benchmark_path)
    train_data = _load_jsonl(train_dataset_path)

    if not benchmark or not train_data:
        return {"leakage_rate": 0.0, "leaked_indices": []}

    # build train set of input texts for comparison
    train_texts = set()
    for example in train_data:
        text = _extract_text(example)
        if text:
            # use 10-char shingles for fast comparison
            train_texts.update(_shingles(text.lower(), k=10))

    leaked = []
    for i, bench_example in enumerate(benchmark):
        input_text = bench_example.get("input", "")
        if not input_text:
            continue

        bench_shingles = _shingles(input_text.lower(), k=10)
        if not bench_shingles:
            continue

        overlap = len(bench_shingles & train_texts) / len(bench_shingles)
        if overlap >= threshold:
            leaked.append(i)

    leakage_rate = len(leaked) / len(benchmark) if benchmark else 0.0

    result = {
        "leakage_rate": leakage_rate,
        "leaked_count": len(leaked),
        "total_benchmark": len(benchmark),
        "leaked_indices": leaked[:20],  # first 20 only
    }

    if leakage_rate > 0.1:
        logger.warning(f"High benchmark leakage: {leakage_rate:.1%}")

    return result



def _extract_text(example: dict) -> str:
    if "input" in example:
        return example["input"]
    if "text" in example:
        return example["text"][:500]
    if "messages" in example:
        msgs = example["messages"]
        for msg in msgs:
            if msg.get("role") == "user":
                return msg.get("content", "")
    if "conversations" in example:
        for turn in example["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            if role in ("human", "user"):
                return turn.get("value", turn.get("content", ""))
    return ""


def _shingles(text: str, k: int = 10) -> set:
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}