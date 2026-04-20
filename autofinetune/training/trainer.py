import os
import torch
from pathlib import Path
from typing import Generator
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import load_dataset
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from autofinetune.graph.state import RunConfig
from autofinetune.training.peft_config import build_lora_config, build_qlora_config


def launch_training(
    run_id: str,
    config: RunConfig,
    dataset_path: str,
    base_model: str,
    training_mode: str,
    checkpoint_dir: str,
) -> Generator[dict, None, None]:
    """
    Launches a finetuning run and yields progress updates.
    Each update is a dict with step, train_loss, eval_loss, checkpoint_path.
    The monitor iterates over these updates to watch the run.
    """
    logger.info(f"Launching training | mode={training_mode} | model={base_model}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(base_model)
    model = _load_model(base_model, training_mode)
    dataset = _load_dataset(dataset_path, tokenizer)
    training_args = _build_training_args(config, checkpoint_dir, dataset)

    # shared update queue between trainer callback and this generator
    updates = []

    class ProgressCallback(TrainerCallback):
        def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
            update = {
                "step": state.global_step,
                "train_loss": metrics.get("train_loss"),
                "eval_loss": metrics.get("eval_loss"),
                "epoch": state.epoch,
                "checkpoint_path": os.path.join(
                    checkpoint_dir,
                    f"checkpoint-{state.global_step}"
                ),
            }
            updates.append(update)
            logger.debug(f"Step {state.global_step} | eval_loss={metrics.get('eval_loss', 'N/A'):.4f}")

        def on_log(self, args, state: TrainerState, control: TrainerControl, logs, **kwargs):
            if "loss" in logs:
                updates.append({
                    "step": state.global_step,
                    "train_loss": logs["loss"],
                    "eval_loss": None,
                    "epoch": state.epoch,
                    "checkpoint_path": None,
                })

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        callbacks=[ProgressCallback()],
        max_seq_length=2048,
    )

    # run training in a thread so we can yield updates
    import threading

    training_error = [None]

    def run_trainer():
        try:
            trainer.train()
            trainer.save_model(checkpoint_dir)
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "tokenizer"))
        except Exception as e:
            training_error[0] = e
            logger.error(f"Training thread error: {e}")

    thread = threading.Thread(target=run_trainer, daemon=True)
    thread.start()

    # yield updates as they come in
    last_yielded = 0
    while thread.is_alive() or last_yielded < len(updates):
        while last_yielded < len(updates):
            yield updates[last_yielded]
            last_yielded += 1
        thread.join(timeout=2.0)

    if training_error[0]:
        raise training_error[0]

    logger.info(f"Training complete for {run_id}")


def _load_tokenizer(base_model: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right padding for training
    return tokenizer


def _load_model(base_model: str, training_mode: str) -> AutoModelForCausalLM:
    if training_mode == "qlora":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        lora_cfg = build_qlora_config()
        model = get_peft_model(model, lora_cfg)

    elif training_mode == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        lora_cfg = build_lora_config()
        model = get_peft_model(model, lora_cfg)

    else:  # full finetuning
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model


def _load_dataset(dataset_path: str, tokenizer: AutoTokenizer) -> dict:
    ext = Path(dataset_path).suffix

    if ext == ".jsonl":
        ds = load_dataset("json", data_files=dataset_path)
    elif ext == ".csv":
        ds = load_dataset("csv", data_files=dataset_path)
    else:
        ds = load_dataset(dataset_path)

    # if no validation split exists, create one
    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=0.05, seed=42)
        ds = {"train": split["train"], "validation": split["test"]}

    logger.info(f"Dataset loaded | train={len(ds['train'])} | val={len(ds['validation'])}")
    return ds


def _build_training_args(
    config: RunConfig,
    checkpoint_dir: str,
    dataset: dict,
) -> TrainingArguments:
    total_samples = len(dataset["train"])
    steps_per_epoch = max(1, total_samples // (config.batch_size * config.gradient_accumulation))
    total_steps = steps_per_epoch * config.epochs

    return TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.scheduler,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",           
        dataloader_num_workers=4,
        remove_unused_columns=False,
        group_by_length=True,       # speeds up training by grouping similar lengths
    )