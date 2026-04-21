from datetime import datetime
from uuid import uuid4
from loguru import logger
from autofinetune.graph.state import ExperimentState, RunResult, RunConfig
from autofinetune.config.loader import ExperimentConfig


# ── node: init ────────────────────────────────────────────────────────────────

def init_node(state: ExperimentState, config: ExperimentConfig, storage) -> dict:
    """
    First node. Sets up folders, timestamps, initial state.
    Idempotent — safe to re-run if we're resuming.
    """
    logger.info(f"Initializing experiment: {state.experiment_id}")

    storage.init_experiment(state.experiment_id)

    return {
        "current_phase": "data_prep",
        "start_time": state.start_time or datetime.now().isoformat(),
    }


# ── node: data_prep ───────────────────────────────────────────────────────────

def data_prep_node(state: ExperimentState, config: ExperimentConfig, data_agent) -> dict:
    """
    Data agent cleans the dataset and generates/validates the benchmark.
    Only runs once at the start of an experiment.
    """
    logger.info("Running data preparation...")

    result = data_agent.run(
        task="Prepare dataset and generate benchmark",
        context={
            "dataset_path": config.dataset.path,
            "dataset_format": config.dataset.format,
            "base_model": config.base_model.name,
            "use_case": state.use_case,
            "benchmark_config": config.eval.benchmark,
        }
    )

    logger.info(f"Data prep complete. Quality score: {result['quality_report'].get('score')}")

    return {
        "current_phase": "planning",
        "processed_dataset_path": result["processed_dataset_path"],
        "benchmark_path": result["benchmark_path"],
        "data_quality_report": result["quality_report"],
    }


# ── node: planning ────────────────────────────────────────────────────────────

def planning_node(state: ExperimentState, config: ExperimentConfig, strategist, memory) -> dict:
    """
    Strategist reads run history and proposes the next experiment config.
    Writes a hypothesis and journal entry.
    """
    logger.info(f"Planning run {state.current_run + 1}/{state.max_runs}...")

    # build memory context — recent runs full, older runs compressed
    history_context = memory.build_context(state.all_runs)

    result = strategist.run(
        task="Propose the next finetuning experiment",
        context={
            "use_case": state.use_case,
            "run_history": history_context,
            "best_run": _format_best_run(state),
            "current_run_number": state.current_run + 1,
            "max_runs": state.max_runs,
            "budget_remaining": state.max_runs - state.current_run,
            "hours_used": state.total_hours_used,
            "hp_space": config.hyperparameter_space,
            "lessons_learned": state.lessons_learned,
            "training_mode": config.training.mode,
        }
    )

    run_id = f"run_{str(state.current_run + 1).zfill(3)}_{uuid4().hex[:6]}"

    logger.info(f"Hypothesis: {result['hypothesis'][:80]}...")

    return {
        "current_phase": "training",
        "current_run_id": run_id,
        "current_config": RunConfig(**result["config"]),
        "current_hypothesis": result["hypothesis"],
    }


# ── node: training ────────────────────────────────────────────────────────────

def training_node(state: ExperimentState, config: ExperimentConfig, monitor, storage) -> dict:
    """
    Launches the training job. Monitor watches it and handles early stopping.
    """
    logger.info(f"Starting training for {state.current_run_id}...")

    checkpoint_dir = str(storage.checkpoint_dir(
        state.experiment_id,
        state.current_run_id
    ))

    training_report = monitor.watch(
        run_id=state.current_run_id,
        config=state.current_config,
        dataset_path=state.processed_dataset_path,
        base_model=config.base_model.name,
        training_mode=config.training.mode,
        checkpoint_dir=checkpoint_dir,
        early_stopping=config.early_stopping,
        eval_every_n_steps=config.eval.eval_every_n_steps,
    )

    # build partial run result
    current_run = RunResult(
        run_id=state.current_run_id,
        config=state.current_config,
        hypothesis=state.current_hypothesis,
        best_checkpoint_path=training_report.get("best_checkpoint_path"),
        loss_curve=training_report.get("loss_curve", []),
        training_report=training_report,
        status="training" if not training_report.get("failed") else "failed",
        failure_reason=training_report.get("failure_reason"),
    )

    # save to disk immediately
    storage.save_run(state.experiment_id, current_run)

    if training_report.get("failed"):
        logger.warning(f"Training failed: {training_report.get('failure_reason')}")
        return {
            "current_phase": "update",
            "all_runs": state.all_runs + [current_run],
        }

    logger.info(f"Training complete. Best checkpoint: {training_report.get('best_checkpoint_path')}")

    return {
        "current_phase": "evaluating",
        "all_runs": state.all_runs + [current_run],
    }


# ── node: eval ────────────────────────────────────────────────────────────────

def eval_node(state: ExperimentState, config: ExperimentConfig, evaluator, storage) -> dict:
    """
    Evaluator scores the checkpoint against the benchmark.
    """
    logger.info(f"Evaluating {state.current_run_id}...")

    # get the current run from state
    current_run = _get_current_run(state)

    if current_run.status == "failed":
        logger.warning("Skipping eval — training failed")
        return {"current_phase": "update"}

    if not current_run.best_checkpoint_path:
        logger.warning("Skipping eval — no checkpoint path available")
        return {"current_phase": "update"}

    eval_report = evaluator.run(
        task="Evaluate this finetuned checkpoint",
        context={
            "checkpoint_path": current_run.best_checkpoint_path,
            "benchmark_path": state.benchmark_path,
            "use_case": state.use_case,
            "primary_metric": config.eval.primary_metric,
            "base_model": config.base_model.name,
        }
    )

    # update the run result with eval scores
    updated_run = current_run.model_copy(update={
        "eval_score": eval_report["primary_score"],
        "eval_breakdown": eval_report["breakdown"],
        "status": "completed",
    })

    # replace in all_runs list
    updated_runs = [
        updated_run if r.run_id == state.current_run_id else r
        for r in state.all_runs
    ]

    storage.save_run(state.experiment_id, updated_run)

    logger.info(f"Eval complete. Score: {eval_report['primary_score']:.4f}")

    return {
        "current_phase": "update",
        "all_runs": updated_runs,
    }


# ── node: update ──────────────────────────────────────────────────────────────

def update_node(state: ExperimentState, config: ExperimentConfig, memory, journal, storage) -> dict:
    """
    After each run — update leaderboard, memory, journal, budget tracking.
    Decides if this run is the new best.
    """
    logger.info(f"Updating state after {state.current_run_id}...")

    current_run = _get_current_run(state)

    # update best run
    new_best_id = state.best_run_id
    new_best_score = state.best_score

    if current_run.eval_score and current_run.eval_score > state.best_score:
        new_best_id = state.current_run_id
        new_best_score = current_run.eval_score
        logger.info(f"New best run: {state.current_run_id} ({new_best_score:.4f})")

    # update leaderboard on disk
    storage.update_leaderboard(state.experiment_id, state)

    # write journal entry
    journal_entry = journal.write_entry(
        run_id=state.current_run_id,
        run=current_run,
        is_best=new_best_id == state.current_run_id,
        run_number=state.current_run + 1,
    )
    storage.save_journal_entry(state.experiment_id, state.current_run_id, journal_entry)

    # compress memory if needed
    new_compressed = memory.maybe_compress(state.all_runs)

    # extract new lessons
    new_lessons = memory.extract_lessons(state.all_runs)

    # cleanup checkpoints — keep only top K
    top_run_ids = _get_top_k_run_ids(state.all_runs, config.storage.save_top_k_checkpoints)
    storage.cleanup_checkpoints(state.experiment_id, top_run_ids)

    # update hours used
    hours_used = _calculate_hours(state.start_time)

    return {
        "current_phase": "planning",
        "current_run": state.current_run + 1,
        "runs_completed": state.runs_completed + 1,
        "best_run_id": new_best_id,
        "best_score": new_best_score,
        "compressed_history": new_compressed,
        "lessons_learned": new_lessons,
        "total_hours_used": hours_used,
        "current_run_id": None,
        "current_config": None,
        "current_hypothesis": None,
    }


# ── node: report ──────────────────────────────────────────────────────────────

def report_node(state: ExperimentState, journal, storage) -> dict:
    """
    Final node. Generates the research report and wraps up.
    """
    logger.info("Generating final report...")

    report = journal.generate_report(
        experiment_id=state.experiment_id,
        use_case=state.use_case,
        all_runs=state.all_runs,
        best_run_id=state.best_run_id,
        best_score=state.best_score,
        lessons_learned=state.lessons_learned,
        total_hours=state.total_hours_used,
    )

    storage.save_report(state.experiment_id, report)

    logger.info(f"Report saved. Best run: {state.best_run_id} ({state.best_score:.4f})")

    return {
        "current_phase": "complete",
        "should_stop": True,
        "stop_reason": state.stop_reason or "experiment_complete",
    }


# ── node: error ───────────────────────────────────────────────────────────────

def error_node(state: ExperimentState, storage) -> dict:
    """
    Called when something unrecoverable happens.
    Saves state so we can inspect what went wrong.
    """
    logger.error(f"Experiment failed: {state.error}")
    storage.save_state(state)

    return {
        "current_phase": "failed",
        "should_stop": True,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_current_run(state: ExperimentState) -> RunResult:
    for run in state.all_runs:
        if run.run_id == state.current_run_id:
            return run
    raise ValueError(f"Run {state.current_run_id} not found in state")


def _format_best_run(state: ExperimentState) -> dict | None:
    if not state.best_run_id:
        return None
    for run in state.all_runs:
        if run.run_id == state.best_run_id:
            return {
                "run_id": run.run_id,
                "score": run.eval_score,
                "config": run.config.model_dump(),
                "hypothesis": run.hypothesis,
            }
    return None


def _get_top_k_run_ids(runs: list[RunResult], k: int) -> list[str]:
    completed = [r for r in runs if r.status == "completed" and r.eval_score]
    ranked = sorted(completed, key=lambda r: r.eval_score, reverse=True)
    return [r.run_id for r in ranked[:k]]


def _calculate_hours(start_time: str | None) -> float:
    if not start_time:
        return 0.0
    start = datetime.fromisoformat(start_time)
    delta = datetime.now() - start
    return delta.total_seconds() / 3600
