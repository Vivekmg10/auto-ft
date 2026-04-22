from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal
import yaml


class DatasetConfig(BaseModel):
    path: str
    format: Literal["sharegpt", "alpaca", "raw", "custom"] = "sharegpt"
    language: str = "en"


class BaseModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True


class HPRangeConfig(BaseModel):
    min: float
    max: float


class HPCategoricalConfig(BaseModel):
    values: list


class HyperparameterSpace(BaseModel):
    learning_rate: HPRangeConfig
    lora_rank: HPCategoricalConfig
    lora_alpha: HPCategoricalConfig
    batch_size: HPCategoricalConfig
    gradient_accumulation: HPCategoricalConfig
    warmup_ratio: HPRangeConfig
    epochs: HPRangeConfig
    scheduler: HPCategoricalConfig


class TrainingConfig(BaseModel):
    mode: Literal["lora", "qlora", "full"] = "lora"
    max_runs: int = 20
    budget_hours: float = 48.0


class EvalConfig(BaseModel):
    benchmark: str = "auto"
    primary_metric: str = "conversation_quality"
    eval_every_n_steps: int = 200
    judge_model: str = "anthropic/claude-sonnet-4-5"


class AgentsConfig(BaseModel):
    strategist: str = "anthropic/claude-opus-4-5"
    data_agent: str = "groq/llama-3.1-70b-versatile"
    monitor: str = "ollama/qwen2.5:7b"
    evaluator: str = "anthropic/claude-sonnet-4-5"


class StorageConfig(BaseModel):
    base_path: str = "./experiments"
    save_top_k_checkpoints: int = 3


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience_steps: int = 400
    min_improvement: float = 0.01


class ExperimentConfig(BaseModel):
    id: str
    use_case: str
    dataset: DatasetConfig
    base_model: BaseModelConfig = Field(default_factory=BaseModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hyperparameter_space: HyperparameterSpace | None = None
    eval: EvalConfig = Field(default_factory=EvalConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    @property
    def experiment_dir(self) -> Path:
        return Path(self.storage.base_path) / self.id


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    experiment_data = raw.get("experiment", {})
    experiment_data["dataset"] = raw.get("dataset", {})
    experiment_data["base_model"] = raw.get("base_model", {})
    experiment_data["training"] = raw.get("training", {})
    experiment_data["hyperparameter_space"] = raw.get("hyperparameter_space")
    experiment_data["eval"] = raw.get("eval", {})
    experiment_data["agents"] = raw.get("agents", {})
    experiment_data["storage"] = raw.get("storage", {})
    experiment_data["early_stopping"] = raw.get("early_stopping", {})

    cfg = ExperimentConfig(**experiment_data)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    errors = []

    if not Path(cfg.dataset.path).exists():
        errors.append(f"dataset.path does not exist: {cfg.dataset.path!r}")

    if cfg.hyperparameter_space:
        hp = cfg.hyperparameter_space
        if hp.learning_rate.min >= hp.learning_rate.max:
            errors.append(
                f"hyperparameter_space.learning_rate: min ({hp.learning_rate.min}) "
                f"must be less than max ({hp.learning_rate.max})"
            )
        if hp.epochs.min >= hp.epochs.max:
            errors.append(
                f"hyperparameter_space.epochs: min ({hp.epochs.min}) "
                f"must be less than max ({hp.epochs.max})"
            )
        if hp.warmup_ratio.min >= hp.warmup_ratio.max:
            errors.append(
                f"hyperparameter_space.warmup_ratio: min ({hp.warmup_ratio.min}) "
                f"must be less than max ({hp.warmup_ratio.max})"
            )
        if not hp.lora_rank.values:
            errors.append("hyperparameter_space.lora_rank.values must not be empty")
        if not hp.scheduler.values:
            errors.append("hyperparameter_space.scheduler.values must not be empty")

    if cfg.training.max_runs < 1:
        errors.append(f"training.max_runs must be >= 1, got {cfg.training.max_runs}")

    if errors:
        raise ValueError("Invalid config:\n" + "\n".join(f"  - {e}" for e in errors))