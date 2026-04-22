import time
import subprocess
from pathlib import Path
from loguru import logger
from autofinetune.agents.base import BaseAgent
from autofinetune.graph.state import RunConfig


MONITOR_PROMPT = """
You are a training monitor for LLM finetuning experiments.
You watch training jobs and make early stopping decisions.

When given a loss curve and training statistics, you decide:
1. Should we stop early? (is the run clearly failing or plateauing?)
2. Is there anything anomalous worth flagging?
3. What is the best checkpoint to use for evaluation?

Be conservative with early stopping — only stop if:
- Loss is clearly diverging (increasing consistently for 200+ steps)
- Loss has completely plateaued with zero improvement for patience_steps
- The run has clearly failed (NaN loss, OOM errors)

Return your decision as JSON.
"""


class TrainingMonitor(BaseAgent):
    def __init__(self, model: str):
        super().__init__(
            model=model,
            system_prompt=MONITOR_PROMPT,
            tools=[],
            temperature=0.1,
            max_iterations=1,
        )

    def watch(
        self,
        run_id: str,
        config: RunConfig,
        dataset_path: str,
        base_model: str,
        training_mode: str,
        checkpoint_dir: str,
        early_stopping,
        eval_every_n_steps: int,
    ) -> dict:
        """
        Launches training and monitors it.
        Returns a training report when done.
        """
        from autofinetune.training.trainer import launch_training

        loss_curve = []
        best_checkpoint = None
        best_eval_loss = float("inf")

        try:
            for update in launch_training(
                run_id=run_id,
                config=config,
                dataset_path=dataset_path,
                base_model=base_model,
                training_mode=training_mode,
                checkpoint_dir=checkpoint_dir,
            ):
                # update is a dict emitted by the trainer at each eval step
                loss_curve.append(update)

                # track best checkpoint
                if update.get("eval_loss", float("inf")) < best_eval_loss:
                    best_eval_loss = update["eval_loss"]
                    best_checkpoint = update.get("checkpoint_path")

                # check early stopping
                if early_stopping.enabled and len(loss_curve) > 3:
                    should_stop = self._check_early_stop(
                        loss_curve,
                        early_stopping.patience_steps,
                        early_stopping.min_improvement,
                    )
                    if should_stop:
                        logger.info(f"Early stopping triggered at step {update.get('step')}")
                        break

            return {
                "run_id": run_id,
                "best_checkpoint_path": best_checkpoint or checkpoint_dir,
                "loss_curve": loss_curve,
                "best_eval_loss": best_eval_loss,
                "steps_completed": loss_curve[-1].get("step", 0) if loss_curve else 0,
                "failed": False,
            }

        except Exception as e:
            logger.error(f"Training failed for {run_id}: {e}")
            return {
                "run_id": run_id,
                "best_checkpoint_path": None,
                "loss_curve": loss_curve,
                "failed": True,
                "failure_reason": str(e),
            }

    def _check_early_stop(
        self,
        loss_curve: list[dict],
        patience_steps: int,
        min_improvement: float,
    ) -> bool:
        """
        Simple early stopping logic.
        LLM is only called for anomaly detection, not for this decision.
        """
        if len(loss_curve) < 5:
            return False

        eval_losses = [
            s["eval_loss"] for s in loss_curve
            if "eval_loss" in s
        ]

        if not eval_losses:
            return False

        # NaN/Inf means training is broken — stop immediately
        import math
        if any(math.isnan(loss) or math.isinf(loss) for loss in eval_losses):
            logger.error("NaN or Inf detected in loss curve — stopping training immediately")
            return True

        # divergence check
        recent = eval_losses[-5:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            logger.warning("Loss diverging — considering early stop")
            return True

        # plateau check
        best_so_far = min(eval_losses[:-1]) if len(eval_losses) > 1 else eval_losses[0]
        recent_best = min(eval_losses[-3:])
        improvement = best_so_far - recent_best

        steps_since_improvement = len(eval_losses) * (patience_steps // len(eval_losses))
        if improvement < min_improvement and steps_since_improvement >= patience_steps:
            return True

        return False

    def _build_prompt(self, task: str, context: dict) -> str:
        return f"""
                    TASK: {task}

                    LOSS CURVE:
                    {context.get('loss_curve', [])}
    
                    TRAINING CONFIG:
                    {context.get('config', {})}

                    Make your decision.
                """

    def _parse_output(self, content: str) -> dict:
        import json, re
        pattern = r"```json\s*(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return {"stop": False, "reason": ""}