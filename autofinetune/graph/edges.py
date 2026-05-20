from autofinetune.graph.state import ExperimentState
from autofinetune.config.loader import ExperimentConfig
from loguru import logger


def should_continue(state: ExperimentState, config: ExperimentConfig) -> str:
    """
    Main decision point after update_node.
    Returns the name of the next node to go to.
    """

    # hard stop flags
    if state.should_stop:
        logger.info(f"Stopping: {state.stop_reason}")
        return "report"

    if state.current_phase == "failed":
        return "error"

    # budget checks
    if state.current_run >= state.max_runs:
        logger.info(f"Reached max runs ({state.max_runs})")
        return "report"

    if state.total_hours_used >= config.training.budget_hours:
        logger.info(f"Reached hour budget ({config.training.budget_hours}h)")
        return "report"

    # convergence check — if last 5 runs show no improvement, stop
    if _is_converged(state):
        logger.info("Convergence detected — no improvement in last 5 runs")
        return "report"

    # all good, keep going
    return "planning"


def skip_eval_if_failed(state: ExperimentState) -> str:
    """
    After training_node — skip eval if training failed.
    """
    from autofinetune.graph.nodes import _get_current_run

    try:
        current_run = _get_current_run(state)
        if current_run.status == "failed":
            logger.warning("Training failed — skipping eval, going to update")
            return "update"
        return "eval"
    except ValueError:
        return "update"


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_converged(state: ExperimentState, window: int = 5, threshold: float = 0.005) -> bool:
    """
    Check if the last N completed runs show no meaningful improvement.
    Uses linear regression slope to detect true plateau vs noise.
    """
    completed = [r for r in state.all_runs if r.status == "completed" and r.eval_score]

    if len(completed) < window:
        return False

    recent = [r.eval_score for r in completed[-window:]]

    # compute simple linear slope
    n = len(recent)
    if n < 2:
        return False

    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = sum(recent) / n

    numerator = sum((x[i] - mean_x) * (recent[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

    if denominator == 0:
        return False

    slope = numerator / denominator

    # converged if slope is near zero (no improvement trend)
    return abs(slope) < threshold