import json
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
from autofinetune.tools.data_cleaning import _load_jsonl, _save_jsonl


def convert_to_chat_template(
    dataset_path: str,
    output_path: str,
    base_model: str,
    input_format: str = "sharegpt",
    system_prompt: str | None = None,
) -> dict:
    """
    Converts any supported format to the correct chat template
    for the target base model.
    """
    examples = _load_jsonl(dataset_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
    except Exception as e:
        return {"error": f"Failed to load tokenizer for {base_model}: {e}"}

    converted = []
    failed = 0

    for example in examples:
        try:
            messages = _to_messages(example, input_format, system_prompt)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            converted.append({
                "text": text,
                "messages": messages,
            })
        except Exception as e:
            failed += 1
            if failed <= 3:
                logger.warning(f"Conversion failed for example: {e}")

    _save_jsonl(converted, output_path)

    result = {
        "original_count": len(examples),
        "converted_count": len(converted),
        "failed_count": failed,
        "output_path": output_path,
        "template_used": _detect_template(base_model),
    }

    logger.info(f"Converted {len(converted)}/{len(examples)} examples to chat template")
    return result


def validate_schema(dataset_path: str, base_model: str) -> dict:
    """
    Validates that a dataset is correctly formatted for training.
    """
    examples = _load_jsonl(dataset_path)

    if not examples:
        return {"valid": False, "error": "Empty dataset"}

    issues = []
    sample = examples[0]

    # check required fields
    has_text = "text" in sample
    has_messages = "messages" in sample

    if not has_text and not has_messages:
        issues.append("Missing 'text' or 'messages' field")

    # check messages format
    if has_messages:
        messages = sample["messages"]
        if not isinstance(messages, list):
            issues.append("'messages' must be a list")
        else:
            for msg in messages[:3]:
                if "role" not in msg:
                    issues.append("Message missing 'role' field")
                if "content" not in msg:
                    issues.append("Message missing 'content' field")

    # check for empty content
    empty_count = 0
    for example in examples[:100]:
        text = example.get("text", "")
        if not text or len(text.strip()) < 5:
            empty_count += 1

    if empty_count > 10:
        issues.append(f"{empty_count} examples have empty or near-empty text")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_examples": len(examples),
        "sample_fields": list(sample.keys()),
    }


def split_dataset(
    dataset_path: str,
    output_dir: str,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Splits dataset into train and validation sets.
    """
    import random
    random.seed(seed)

    examples = _load_jsonl(dataset_path)
    random.shuffle(examples)

    val_size = max(1, int(len(examples) * val_ratio))
    val_set = examples[:val_size]
    train_set = examples[val_size:]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_path = str(output / "train.jsonl")
    val_path = str(output / "val.jsonl")

    _save_jsonl(train_set, train_path)
    _save_jsonl(val_set, val_path)

    result = {
        "train_count": len(train_set),
        "val_count": len(val_set),
        "train_path": train_path,
        "val_path": val_path,
    }

    logger.info(f"Split: {len(train_set)} train, {len(val_set)} val")
    return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_messages(
    example: dict,
    input_format: str,
    system_prompt: str | None,
) -> list[dict]:
    """
    Convert any format to a list of {"role": ..., "content": ...} messages.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if input_format == "sharegpt" or "conversations" in example:
        for turn in example.get("conversations", []):
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            role = _normalize_role(role)
            messages.append({"role": role, "content": content})

    elif input_format == "messages" or "messages" in example:
        for msg in example["messages"]:
            messages.append({
                "role": _normalize_role(msg["role"]),
                "content": msg["content"],
            })

    elif input_format == "alpaca" or "instruction" in example:
        instruction = example.get("instruction", "")
        context = example.get("input", example.get("context", ""))
        output = example.get("output", example.get("response", ""))

        user_content = f"{instruction}\n\n{context}".strip() if context else instruction
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})

    elif "prompt" in example and "completion" in example:
        messages.append({"role": "user", "content": example["prompt"]})
        messages.append({"role": "assistant", "content": example["completion"]})

    elif "input" in example and "output" in example:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})

    else:
        raise ValueError(f"Cannot convert format: {list(example.keys())}")

    return messages


def _normalize_role(role: str) -> str:
    role = role.lower().strip()
    if role in ("human", "user", "question"):
        return "user"
    if role in ("gpt", "assistant", "bot", "answer", "model"):
        return "assistant"
    if role in ("system", "sys"):
        return "system"
    return role


def _detect_template(base_model: str) -> str:
    model_lower = base_model.lower()
    if "qwen" in model_lower:
        return "chatml"
    if "llama" in model_lower:
        return "llama3"
    if "mistral" in model_lower:
        return "mistral"
    if "gemma" in model_lower:
        return "gemma"
    if "phi" in model_lower:
        return "phi3"
    return "unknown"