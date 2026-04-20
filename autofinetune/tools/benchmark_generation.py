import json
from pathlib import Path
from loguru import logger
from litellm import completion
from autofinetune.tools.data_cleaning import _load_jsonl, _save_jsonl


def generate_benchmark_from_description(
    use_case: str,
    output_path: str,
    n_examples: int = 50,
    difficulty_distribution: dict | None = None,
) -> dict:
    """
    Generates a benchmark eval set from a use case description.
    Pipeline: decompose → generate per dimension → validate → save
    """
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

    logger.info(f"Generating benchmark: {n_examples} examples for '{use_case[:50]}...'")

    # step 1 — decompose use case into capability dimensions
    dimensions = _decompose_use_case(use_case)
    logger.debug(f"Dimensions: {dimensions}")

    # step 2 — generate examples per dimension
    examples = []
    per_dimension = max(1, n_examples // len(dimensions))

    for dimension in dimensions:
        dim_examples = _generate_dimension_examples(
            use_case=use_case,
            dimension=dimension,
            n=per_dimension,
            difficulty_distribution=difficulty_distribution,
        )
        examples.extend(dim_examples)

    # trim to exact count
    examples = examples[:n_examples]

    # step 3 — save
    _save_jsonl(examples, output_path)

    result = {
        "n_generated": len(examples),
        "dimensions": dimensions,
        "output_path": output_path,
    }

    logger.info(f"Generated {len(examples)} benchmark examples")
    return result


def generate_benchmark_from_dataset(
    dataset_path: str,
    output_path: str,
    n_examples: int = 50,
) -> dict:
    """
    Generates benchmark by sampling training examples and
    transforming them into eval format with rubrics.
    """
    import random
    examples = _load_jsonl(dataset_path)
    random.shuffle(examples)
    sample = examples[:min(n_examples * 2, len(examples))]

    benchmark = []
    for example in sample[:n_examples]:
        bench_example = _transform_to_benchmark(example)
        if bench_example:
            benchmark.append(bench_example)

    _save_jsonl(benchmark, output_path)

    result = {
        "sampled_from": len(examples),
        "n_generated": len(benchmark),
        "output_path": output_path,
    }

    logger.info(f"Generated {len(benchmark)} benchmark examples from dataset")
    return result


def validate_benchmark_quality(
    benchmark_path: str,
    train_dataset_path: str,
) -> dict:
    """
    Validates benchmark for leakage, ambiguity, and coverage.
    """
    from autofinetune.tools.validation import check_leakage

    benchmark = _load_jsonl(benchmark_path)

    if not benchmark:
        return {"valid": False, "error": "Empty benchmark"}

    # check leakage
    leakage_report = check_leakage(
        benchmark_path=benchmark_path,
        train_dataset_path=train_dataset_path,
    )

    # check difficulty distribution
    difficulties = [e.get("difficulty", "unknown") for e in benchmark]
    difficulty_counts = {
        "easy": difficulties.count("easy"),
        "medium": difficulties.count("medium"),
        "hard": difficulties.count("hard"),
        "unknown": difficulties.count("unknown"),
    }

    # check rubric coverage
    has_rubric = sum(1 for e in benchmark if e.get("rubric"))
    rubric_coverage = has_rubric / len(benchmark)

    issues = []
    if leakage_report.get("leakage_rate", 0) > 0.1:
        issues.append(f"High leakage rate: {leakage_report['leakage_rate']:.1%}")
    if rubric_coverage < 0.8:
        issues.append(f"Low rubric coverage: {rubric_coverage:.1%}")
    if len(benchmark) < 10:
        issues.append("Benchmark too small — less than 10 examples")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_examples": len(benchmark),
        "difficulty_distribution": difficulty_counts,
        "rubric_coverage": rubric_coverage,
        "leakage_report": leakage_report,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _decompose_use_case(use_case: str) -> list[str]:
    """
    Uses LLM to decompose a use case into testable capability dimensions.
    """
    prompt = f"""
You are designing an evaluation benchmark for an LLM.

USE CASE: {use_case}

List 5-7 specific capability dimensions that should be evaluated.
Each dimension should be concrete and testable.

Return ONLY a JSON array of strings like:
["dimension 1", "dimension 2", ...]

No explanation, just the array.
"""
    try:
        response = completion(
            model="groq/llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        dimensions = json.loads(content)
        if isinstance(dimensions, list):
            return dimensions[:7]
    except Exception as e:
        logger.warning(f"Dimension decomposition failed: {e}")

    # fallback dimensions
    return [
        "task completion",
        "response quality",
        "tone appropriateness",
        "accuracy",
        "edge case handling",
    ]


def _generate_dimension_examples(
    use_case: str,
    dimension: str,
    n: int,
    difficulty_distribution: dict,
) -> list[dict]:
    """
    Generates benchmark examples for a specific capability dimension.
    """
    n_easy = max(1, int(n * difficulty_distribution.get("easy", 0.3)))
    n_medium = max(1, int(n * difficulty_distribution.get("medium", 0.5)))
    n_hard = max(1, n - n_easy - n_medium)

    prompt = f"""
Generate {n} evaluation examples for testing this capability:

USE CASE: {use_case}
CAPABILITY DIMENSION: {dimension}

Generate exactly {n_easy} easy, {n_medium} medium, and {n_hard} hard examples.

Each example must have:
- "input": the user message/prompt to test
- "ideal": what a perfect response would contain (key points, not exact wording)
- "rubric": specific criteria for judging the response (2-3 sentences)
- "difficulty": "easy", "medium", or "hard"
- "dimension": "{dimension}"

Return ONLY a valid JSON array. No explanation.
"""

    try:
        response = completion(
            model="groq/llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
        )
        content = response.choices[0].message.content.strip()

        # extract JSON array
        import re
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            examples = json.loads(match.group(0))
            if isinstance(examples, list):
                return examples
    except Exception as e:
        logger.warning(f"Example generation failed for dimension '{dimension}': {e}")

    return []


def _transform_to_benchmark(example: dict) -> dict | None:
    """
    Transform a training example into benchmark format with rubric.
    """
    # extract input
    input_text = None
    ideal = None

    if "messages" in example:
        messages = example["messages"]
        for msg in messages:
            if msg["role"] == "user" and not input_text:
                input_text = msg["content"]
            if msg["role"] == "assistant" and not ideal:
                ideal = msg["content"]

    elif "conversations" in example:
        for turn in example["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user") and not input_text:
                input_text = content
            if role in ("gpt", "assistant") and not ideal:
                ideal = content

    elif "input" in example:
        input_text = example["input"]
        ideal = example.get("output", "")

    if not input_text or not ideal:
        return None

    return {
        "input": input_text,
        "ideal": ideal[:500],  # truncate ideal to avoid huge context
        "rubric": "Response should be accurate, helpful, and appropriately concise.",
        "difficulty": "medium",
        "dimension": "general",
        "source": "training_sample",
    }
