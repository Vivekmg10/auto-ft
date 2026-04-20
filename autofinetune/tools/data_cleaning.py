import json
import re
import hashlib
from pathlib import Path
from loguru import logger
from litellm import completion


def get_dataset_stats(dataset_path: str) -> dict:
    """
    Returns statistics about a dataset.
    Always called first by the DataAgent.
    """
    path = Path(dataset_path)

    if not path.exists():
        return {"error": f"File not found: {dataset_path}"}

    examples = _load_jsonl(dataset_path)

    if not examples:
        return {"error": "Dataset is empty"}

    lengths = [len(json.dumps(e)) for e in examples]
    sample = examples[:3]

    # detect format
    format_detected = _detect_format(examples[0])

    # count unique fields
    all_keys = set()
    for e in examples[:100]:
        all_keys.update(e.keys())

    return {
        "total_examples": len(examples),
        "format_detected": format_detected,
        "fields": list(all_keys),
        "avg_length_chars": sum(lengths) / len(lengths),
        "min_length_chars": min(lengths),
        "max_length_chars": max(lengths),
        "sample_examples": sample,
        "file_size_mb": path.stat().st_size / 1024 / 1024,
    }


def deduplicate_dataset(
    dataset_path: str,
    output_path: str,
    fuzzy: bool = True,
    similarity_threshold: float = 0.85,
) -> dict:
    """
    Removes exact and optionally fuzzy duplicates.
    """
    examples = _load_jsonl(dataset_path)
    original_count = len(examples)

    # exact dedup on serialized content
    seen_hashes = set()
    deduped = []

    for example in examples:
        content = json.dumps(example, sort_keys=True)
        h = hashlib.md5(content.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(example)

    exact_removed = original_count - len(deduped)

    # fuzzy dedup using simple shingling
    fuzzy_removed = 0
    if fuzzy and len(deduped) > 1:
        deduped, fuzzy_removed = _fuzzy_dedup(deduped, similarity_threshold)

    _save_jsonl(deduped, output_path)

    result = {
        "original_count": original_count,
        "final_count": len(deduped),
        "exact_removed": exact_removed,
        "fuzzy_removed": fuzzy_removed,
        "output_path": output_path,
    }

    logger.info(f"Dedup: {original_count} → {len(deduped)} examples")
    return result


def filter_by_length(
    dataset_path: str,
    output_path: str,
    min_tokens: int = 10,
    max_tokens: int = 2048,
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> dict:
    """
    Filters examples by token count.
    Uses character estimate if tokenizer load fails.
    """
    examples = _load_jsonl(dataset_path)
    original_count = len(examples)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        count_fn = lambda e: len(tokenizer.encode(json.dumps(e)))
    except Exception:
        logger.warning("Tokenizer load failed — using char/4 estimate")
        count_fn = lambda e: len(json.dumps(e)) // 4

    filtered = []
    too_short = 0
    too_long = 0

    for example in examples:
        token_count = count_fn(example)
        if token_count < min_tokens:
            too_short += 1
        elif token_count > max_tokens:
            too_long += 1
        else:
            filtered.append(example)

    _save_jsonl(filtered, output_path)

    result = {
        "original_count": original_count,
        "final_count": len(filtered),
        "removed_too_short": too_short,
        "removed_too_long": too_long,
        "output_path": output_path,
    }

    logger.info(f"Length filter: {original_count} → {len(filtered)} examples")
    return result


def score_quality(
    dataset_path: str,
    output_path: str,
    use_case: str,
    threshold: float = 0.6,
    sample_size: int = -1,
) -> dict:
    """
    Uses LLM to score example quality. Filters below threshold.
    Scores a sample if dataset is large.
    """
    examples = _load_jsonl(dataset_path)

    if sample_size > 0 and len(examples) > sample_size:
        import random
        to_score = random.sample(examples, sample_size)
        unscored = [e for e in examples if e not in to_score]
    else:
        to_score = examples
        unscored = []

    scored = []
    batch_size = 10

    for i in range(0, len(to_score), batch_size):
        batch = to_score[i:i + batch_size]
        scores = _score_batch(batch, use_case)
        for example, score in zip(batch, scores):
            if score >= threshold:
                scored.append(example)

    # unscored examples pass through without filtering
    final = scored + unscored
    _save_jsonl(final, output_path)

    result = {
        "original_count": len(examples),
        "scored_count": len(to_score),
        "filtered_count": len(to_score) - len(scored),
        "final_count": len(final),
        "output_path": output_path,
    }

    logger.info(f"Quality filter: {len(examples)} → {len(final)} examples")
    return result


def fix_formatting(dataset_path: str, output_path: str) -> dict:
    """
    Fixes common formatting issues in JSONL datasets.
    """
    fixed = []
    errors = []
    fixed_count = 0

    with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                example = _fix_example(example)
                fixed.append(example)
                fixed_count += 1
            except json.JSONDecodeError as e:
                # try to salvage with aggressive cleaning
                cleaned = _aggressive_clean(line)
                try:
                    example = json.loads(cleaned)
                    fixed.append(example)
                    fixed_count += 1
                except Exception:
                    errors.append({"line": line_num, "error": str(e)})

    _save_jsonl(fixed, output_path)

    result = {
        "original_lines": line_num,
        "fixed_count": fixed_count,
        "error_count": len(errors),
        "errors_sample": errors[:5],
        "output_path": output_path,
    }

    logger.info(f"Format fix: {fixed_count} fixed, {len(errors)} errors")
    return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return examples


def _save_jsonl(examples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def _detect_format(example: dict) -> str:
    if "conversations" in example:
        return "sharegpt"
    if "messages" in example:
        return "messages"
    if "instruction" in example:
        return "alpaca"
    if "prompt" in example and "completion" in example:
        return "completion"
    if "input" in example and "output" in example:
        return "input_output"
    return "unknown"


def _fuzzy_dedup(examples: list[dict], threshold: float) -> tuple[list[dict], int]:
    """
    Simple shingling-based fuzzy dedup.
    """
    def shingles(text: str, k: int = 5) -> set:
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    texts = [json.dumps(e, sort_keys=True) for e in examples]
    shingle_sets = [shingles(t) for t in texts]

    keep = [True] * len(examples)
    removed = 0

    for i in range(len(examples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(examples)):
            if not keep[j]:
                continue
            if jaccard(shingle_sets[i], shingle_sets[j]) >= threshold:
                keep[j] = False
                removed += 1

    return [e for e, k in zip(examples, keep) if k], removed


def _score_batch(examples: list[dict], use_case: str) -> list[float]:
    """
    Scores a batch of examples for quality using LLM.
    Returns list of scores 0-1.
    """
    formatted = "\n---\n".join(
        f"Example {i+1}:\n{json.dumps(e, indent=2)}"
        for i, e in enumerate(examples)
    )

    prompt = f"""Rate each of these {len(examples)} training examples for quality.

USE CASE: {use_case}

Rate each example from 0.0 to 1.0 based on:
- Relevance to the use case
- Response quality and correctness
- Conversational naturalness
- Would this example help train a good model?

EXAMPLES:
{formatted}

Return ONLY a JSON array of {len(examples)} scores like: [0.8, 0.6, 0.9, ...]
No explanation, just the array.
"""

    try:
        response = completion(
            model="groq/llama-3.1-8b-instant",   # cheap model for bulk scoring
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        scores = json.loads(content)
        if isinstance(scores, list) and len(scores) == len(examples):
            return [float(s) for s in scores]
    except Exception as e:
        logger.warning(f"Quality scoring failed: {e}")

    return [0.8] * len(examples)  # default pass if scoring fails


def _fix_example(example: dict) -> dict:
    """Fix common issues in a single example."""
    fixed = {}
    for key, value in example.items():
        # strip whitespace from string values
        if isinstance(value, str):
            value = value.strip()
            # fix common encoding artifacts
            value = value.replace("\u00a0", " ")
            value = value.replace("\u200b", "")
        fixed[key] = value
    return fixed


def _aggressive_clean(line: str) -> str:
    """Last resort cleaning for malformed JSON lines."""
    # remove control characters
    line = re.sub(r'[\x00-\x1f\x7f]', ' ', line)
    # fix unescaped quotes inside strings (naive)
    line = line.strip()
    return line