import json
from loguru import logger
from litellm import completion
from autofinetune.tools.data_cleaning import _load_jsonl, _save_jsonl


def paraphrase_examples(
    dataset_path: str,
    output_path: str,
    n_variants: int = 1,
) -> dict:
    """
    Rewrites examples to increase dataset diversity.
    Generates n_variants paraphrases per example.
    """
    examples = _load_jsonl(dataset_path)
    augmented = list(examples)  # keep originals

    for example in examples[:100]:   # cap to avoid huge cost
        for _ in range(n_variants):
            variant = _paraphrase_example(example)
            if variant:
                augmented.append(variant)

    _save_jsonl(augmented, output_path)

    return {
        "original_count": len(examples),
        "augmented_count": len(augmented),
        "output_path": output_path,
    }


def generate_synthetic_examples(
    dataset_path: str,
    output_path: str,
    n_examples: int = 100,
    use_case: str = "",
) -> dict:
    """
    Generates new synthetic examples similar to the seed dataset.
    """
    examples = _load_jsonl(dataset_path)
    seed_sample = examples[:10]   # use first 10 as seeds

    synthetic = _generate_synthetic(seed_sample, n_examples, use_case)
    all_examples = examples + synthetic
    _save_jsonl(all_examples, output_path)

    return {
        "original_count": len(examples),
        "synthetic_count": len(synthetic),
        "total_count": len(all_examples),
        "output_path": output_path,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _paraphrase_example(example: dict) -> dict | None:
    try:
        prompt = f"""
Paraphrase this training example. Keep the same meaning and structure
but use different wording. Return valid JSON only.

Original:
{json.dumps(example, indent=2)}

Return the paraphrased version as JSON. No explanation.
"""
        response = completion(
            model="groq/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        content = response.choices[0].message.content.strip()
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.debug(f"Paraphrase failed: {e}")
    return None


def _generate_synthetic(
    seed_examples: list[dict],
    n: int,
    use_case: str,
) -> list[dict]:
    prompt = f"""
Generate {n} new training examples similar to these seed examples.

USE CASE: {use_case}

SEED EXAMPLES:
{json.dumps(seed_examples, indent=2)}

Generate diverse variations. Return a JSON array of {n} examples.
Match the exact same JSON structure as the seeds. No explanation.
"""
    try:
        response = completion(
            model="groq/llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=8000,
        )
        content = response.choices[0].message.content.strip()
        import re
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
    except Exception as e:
        logger.warning(f"Synthetic generation failed: {e}")
    return []