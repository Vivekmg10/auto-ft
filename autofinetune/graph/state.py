from pydantic import BaseModel, Field
from typing import Literal
from dataclasses import dataclass, field


class RunConfig(BaseModel):
    learning_rate: float
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    batch_size: int = 4
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    epochs: int = 3
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0


class RunResult(BaseModel):
    run_id: str
    config: RunConfig
    hypothesis: str
    eval_score: float | None = None
    eval_breakdown: dict = Field(default_factory=dict)
    best_checkpoint_path: str | None = None
    loss_curve: list[dict] = Field(default_factory=list)
    training_report: dict = Field(default_factory=dict)
    status: Literal["pending", "training", "evaluating", "completed", "failed", "pruned"] = "pending"
    failure_reason: str | None = None


class ExperimentState(BaseModel):
    # experiment identity
    experiment_id: str
    use_case: str

    # progress
    current_run: int = 0
    max_runs: int = 20
    current_phase: Literal[
        "init",
        "data_prep",
        "planning",
        "training",
        "evaluating",
        "complete",
        "failed"
    ] = "init"

    # data
    processed_dataset_path: str | None = None
    benchmark_path: str | None = None
    data_quality_report: dict = Field(default_factory=dict)

    # current run
    current_run_id: str | None = None
    current_config: RunConfig | None = None
    current_hypothesis: str | None = None

    # history — this is what Strategist reads
    all_runs: list[RunResult] = Field(default_factory=list)
    best_run_id: str | None = None
    best_score: float = 0.0

    # memory
    compressed_history: str | None = None  # summarized older runs
    lessons_learned: list[str] = Field(default_factory=list)

    # budget tracking
    runs_completed: int = 0
    start_time: str | None = None
    total_hours_used: float = 0.0

    # flags
    should_stop: bool = False
    stop_reason: str | None = None
    error: str | None = None