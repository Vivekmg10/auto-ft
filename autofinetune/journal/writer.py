from datetime import datetime
from loguru import logger
from litellm import completion
from autofinetune.graph.state import RunResult


class JournalWriter:
    """
    Writes research journal entries after each run
    and generates the final experiment report.
    """

    def __init__(self, model: str):
        self.model = model

    def write_entry(
        self,
        run_id: str,
        run: RunResult,
        is_best: bool,
        run_number: int,
    ) -> str:
        """
        Writes a first-person journal entry for a completed run.
        """
        prompt = f"""
You are an ML researcher writing in your research journal.
Write a journal entry for this finetuning experiment run.

Write in first person, past tense. Be honest and analytical.
Note what you expected, what actually happened, and what it means.
Keep it to 3-4 paragraphs.

RUN DETAILS:
- Run ID: {run_id}
- Run Number: {run_number}
- Status: {run.status}
- Hypothesis: {run.hypothesis}
- Config: lr={run.config.learning_rate}, rank={run.config.lora_rank}, epochs={run.config.epochs}, scheduler={run.config.scheduler}
- Eval Score: {run.eval_score if run.eval_score else 'N/A'}
- Eval Breakdown: {run.eval_breakdown}
- Is New Best: {is_best}
- Failure Reason: {run.failure_reason or 'None'}
- Loss Curve Summary: {self._summarize_loss_curve(run.loss_curve)}

Write the journal entry now.
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            entry_body = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Journal entry generation failed: {e}")
            entry_body = self._fallback_entry(run, is_best)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        is_best_badge = " ⭐ NEW BEST" if is_best else ""

        entry = f"""# Run {run_number}: {run_id}{is_best_badge}
**Date**: {timestamp}
**Status**: {run.status.upper()}
**Score**: {run.eval_score:.4f if run.eval_score else 'N/A'}
**Config**: lr={run.config.learning_rate} | rank={run.config.lora_rank} | epochs={run.config.epochs} | scheduler={run.config.scheduler}

---

{entry_body}

---

**Eval Breakdown**:
{self._format_breakdown(run.eval_breakdown)}
"""
        return entry

    def generate_report(
        self,
        experiment_id: str,
        use_case: str,
        all_runs: list[RunResult],
        best_run_id: str | None,
        best_score: float,
        lessons_learned: list[str],
        total_hours: float,
    ) -> str:
        """
        Generates the final research report as markdown.
        """
        completed = [r for r in all_runs if r.status == "completed"]
        failed = [r for r in all_runs if r.status == "failed"]
        best_run = next((r for r in all_runs if r.run_id == best_run_id), None)

        prompt = f"""
Write a research report for this LLM finetuning experiment.

The report should read like a mini research paper — clear, analytical, and useful
for someone who wants to finetune a model for the same task in the future.

EXPERIMENT: {experiment_id}
USE CASE: {use_case}
TOTAL RUNS: {len(all_runs)} ({len(completed)} completed, {len(failed)} failed)
TOTAL HOURS: {total_hours:.1f}h
BEST SCORE: {best_score:.4f}

BEST RUN CONFIG:
{best_run.config.model_dump() if best_run else 'N/A'}

BEST RUN HYPOTHESIS:
{best_run.hypothesis if best_run else 'N/A'}

ALL RUN SCORES:
{self._format_all_scores(all_runs)}

KEY LESSONS LEARNED:
{chr(10).join(f'- {l}' for l in lessons_learned)}

Write the full report with these sections:
1. Executive Summary
2. Experiment Setup
3. Key Findings
4. Hyperparameter Analysis
5. Best Configuration
6. Recommendations for Further Improvement
7. Conclusion

Be specific with numbers. Cite specific runs when making claims.
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=3000,
            )
            report_body = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            report_body = "Report generation failed. See individual run journals."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f"""# AutoFineTune Experiment Report
**Experiment**: {experiment_id}
**Generated**: {timestamp}
**Best Score**: {best_score:.4f}
**Total Runs**: {len(all_runs)}
**Total Hours**: {total_hours:.1f}h

---

{report_body}

---

## Full Leaderboard

| Rank | Run ID | Score | LR | Rank | Epochs | Scheduler |
|------|--------|-------|----|------|--------|-----------|
{self._format_leaderboard_table(all_runs)}
"""

    # ── private ───────────────────────────────────────────────────────────────

    def _summarize_loss_curve(self, loss_curve: list[dict]) -> str:
        if not loss_curve:
            return "No loss data"
        eval_losses = [s["eval_loss"] for s in loss_curve if s.get("eval_loss")]
        if not eval_losses:
            return "No eval loss data"
        return (
            f"start={eval_losses[0]:.4f}, "
            f"end={eval_losses[-1]:.4f}, "
            f"best={min(eval_losses):.4f}, "
            f"steps={len(loss_curve)}"
        )

    def _format_breakdown(self, breakdown: dict) -> str:
        if not breakdown:
            return "No breakdown available"
        return "\n".join(f"- {k}: {v:.4f}" for k, v in breakdown.items())

    def _format_all_scores(self, runs: list[RunResult]) -> str:
        lines = []
        for r in runs:
            score = f"{r.eval_score:.4f}" if r.eval_score else r.status
            lines.append(f"{r.run_id}: {score}")
        return "\n".join(lines)

    def _format_leaderboard_table(self, runs: list[RunResult]) -> str:
        completed = [r for r in runs if r.status == "completed" and r.eval_score]
        ranked = sorted(completed, key=lambda r: r.eval_score, reverse=True)
        rows = []
        for i, r in enumerate(ranked):
            rows.append(
                f"| {i+1} | {r.run_id} | {r.eval_score:.4f} | "
                f"{r.config.learning_rate} | {r.config.lora_rank} | "
                f"{r.config.epochs} | {r.config.scheduler} |"
            )
        return "\n".join(rows)

    def _fallback_entry(self, run: RunResult, is_best: bool) -> str:
        return (
            f"Run completed with status: {run.status}. "
            f"Score: {run.eval_score if run.eval_score else 'N/A'}. "
            f"{'This is the new best run.' if is_best else ''}"
        )