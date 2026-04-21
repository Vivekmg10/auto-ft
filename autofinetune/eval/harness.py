import torch
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def run_checkpoint_on_examples(
    checkpoint_path: str,
    examples: list[dict],
    base_model: str,
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> list[str]:
    """
    Loads a finetuned checkpoint and runs it on benchmark examples.
    Returns a list of model outputs in the same order as examples.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    tokenizer = _load_tokenizer(checkpoint_path, base_model)
    model = _load_model(checkpoint_path, base_model)

    outputs = []
    batches = _batch(examples, batch_size)

    for i, batch in enumerate(batches):
        logger.debug(f"Running batch {i + 1}/{len(batches)}")
        batch_outputs = _run_batch(
            model=model,
            tokenizer=tokenizer,
            examples=batch,
            max_new_tokens=max_new_tokens,
        )
        outputs.extend(batch_outputs)

    # clean up — free VRAM before training starts
    del model
    torch.cuda.empty_cache()

    logger.info(f"Ran {len(outputs)} examples through checkpoint")
    return outputs


def _load_tokenizer(checkpoint_path: str, base_model: str) -> AutoTokenizer:
    """
    Try loading tokenizer from checkpoint first,
    fall back to base model if not saved there.
    """
    checkpoint = Path(checkpoint_path)
    tokenizer_path = checkpoint / "tokenizer"

    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    elif (checkpoint / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        logger.info("No tokenizer in checkpoint, loading from base model")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"  # for batch generation
    return tokenizer


def _load_model(checkpoint_path: str, base_model: str) -> AutoModelForCausalLM:
    """
    Detects whether checkpoint is a full finetune or LoRA adapter
    and loads accordingly.
    """
    checkpoint = Path(checkpoint_path)
    is_lora = (checkpoint / "adapter_config.json").exists()

    if is_lora:
        logger.info("Loading LoRA adapter on top of base model")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        logger.info("Loading full finetuned model")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model


def _run_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    max_new_tokens: int,
) -> list[str]:
    """
    Runs a single batch of examples through the model.
    """
    prompts = [_extract_prompt(e, tokenizer) for e in examples]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy for eval — deterministic
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens, not the input prompt
    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_length:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return [d.strip() for d in decoded]


def _extract_prompt(example: dict, tokenizer: AutoTokenizer) -> str:
    """
    Extracts the prompt from an example dict.
    Handles multiple common formats.
    """
    # chat format with messages list
    if "messages" in example:
        messages = example["messages"]
        # remove the last assistant turn if present — that's what we're generating
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # simple input/output format
    if "input" in example:
        return example["input"]

    # prompt key
    if "prompt" in example:
        return example["prompt"]

    # instruction format
    if "instruction" in example:
        context = example.get("context", "")
        instruction = example["instruction"]
        if context:
            return f"{instruction}\n\nContext: {context}"
        return instruction

    raise ValueError(f"Cannot extract prompt from example: {list(example.keys())}")


def _batch(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def evaluate_with_judge(
    checkpoint_path: str,
    benchmark_path: str,
    base_model: str,
    judge_model: str,
    use_case: str,
) -> dict:
    """
    Full eval pipeline:
    1. Load checkpoint
    2. Run on benchmark
    3. Score with LLM judge
    4. Return aggregated report
    """
    import jsonlines
    from autofinetune.eval.judge import LLMJudge

    # load benchmark
    examples = []
    with jsonlines.open(benchmark_path) as reader:
        for item in reader:
            examples.append(item)

    examples = examples[:50]  # cap at 50

    # run model
    logger.info(f"Running {len(examples)} benchmark examples...")
    outputs = run_checkpoint_on_examples(
        checkpoint_path=checkpoint_path,
        examples=examples,
        base_model=base_model,
    )

    # score with judge
    logger.info("Scoring with LLM judge...")
    judge = LLMJudge(model=judge_model)
    scored = judge.score_batch(
        examples=examples,
        outputs=outputs,
        use_case=use_case,
    )

    # attach inputs and outputs to scored results for report
    for i, score in enumerate(scored):
        score["input"] = examples[i].get("input", "")[:200]
        score["output"] = outputs[i][:200]

    # aggregate
    report = judge.aggregate_scores(scored)
    report["n_examples"] = len(examples)

    return report