from peft import LoraConfig, TaskType
from autofinetune.graph.state import RunConfig


# default target modules for Qwen2.5
# loaded from models.yaml at runtime but these are the defaults
QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


def build_lora_config(
    run_config: RunConfig | None = None,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    rank = (run_config.lora_rank if run_config and run_config.lora_rank else None) or 16
    alpha = (run_config.lora_alpha if run_config and run_config.lora_alpha else None) or 32
    dropout = (run_config.lora_dropout if run_config and run_config.lora_dropout is not None else None)
    if dropout is None:
        dropout = 0.05

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or QWEN_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )


def build_qlora_config(
    run_config: RunConfig | None = None,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    # QLoRA uses same LoRA config — quantization is handled at model load time
    return build_lora_config(run_config, target_modules)