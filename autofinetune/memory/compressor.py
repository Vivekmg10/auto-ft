import json
from loguru import logger
from litellm import completion
from autofinetune.graph.state import RunResult


class MemoryCompressor:
    """
    Manages what goes into the Strategist's context window.

    Two level memory:
    - Recent runs (last N) go in full detail
    - Older runs get compressed into a summary
    - Lessons are extracted as bullet points
    """

    def __init__(self, model: str, max_recent: int = 5):
        self.model = model
        self.max_recent = max_recent

    def build_context(self, all_runs: list[RunResult]) -> str:
        """
        Build the run history string that goes into the Strategist prompt.
        """
        if not all_runs:
            return "No runs yet — this is the first experiment."

        completed = [r for r in all_runs if r.status in ("completed", "failed", "pruned")]

        if len(completed) <= self.max_recent:
            return self._format_runs_full(completed)

        recent = completed[-self.max_recent:]
        older = completed[:-self.max_recent]

        compressed = self._compress(older)

        return f"""SUMMARY OF EARLIER RUNS ({len(older)} runs):
{compressed}

RECENT RUNS (full detail):
{self._format_runs_full(recent)}"""

    def maybe_compress(self, all_runs: list[RunResult]) -> str | None:
        """
        Called after every run update.
        Returns new compressed summary if compression is needed.
        """
        completed = [r for r in all_runs if r.status == "completed"]

        if len(completed) <= self.max_recent:
            return None

        older = completed[:-self.max_recent]
        return self._compress(older)

    def extract_lessons(self, all_runs: list[RunResult]) -> list[str]:
        """
        Extracts key lessons from run history as bullet points.
        Called after each run to keep lessons current.
        """
        completed = [r for r in all_runs if r.status == "completed" and r.eval_score]

        if len(completed) < 3:
            return []

        prompt = f"""
Analyze these finetuning experiment results and extract 3-5 key lessons learned.

Each lesson should be:
- Specific and actionable (e.g. "Learning rates above 3e-4 consistently overfit")
- Grounded in the actual results
- Useful for deciding what to try next

RUNS:
{json.dumps([{
    "run_id": r.run_id,
    "config": r.config.model_dump(),
    "eval_score": r.eval_score,
    "hypothesis": r.hypothesis,
    "status": r.status,
} for r in completed], indent=2)}

Return ONLY a JSON array of lesson strings.
No explanation, just the array.
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            import re
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                lessons = json.loads(match.group(0))
                if isinstance(lessons, list):
                    return [str(l) for l in lessons[:5]]
        except Exception as e:
            logger.warning(f"Lesson extraction failed: {e}")

        return []

    # ── private ───────────────────────────────────────────────────────────────

    def _compress(self, runs: list[RunResult]) -> str:
        """
        Compresses a list of older runs into a concise summary.
        """
        prompt = f"""
Summarize these {len(runs)} finetuning experiment runs concisely.

Focus on:
- What hyperparameter ranges were explored
- What worked well and what didn't
- Any clear patterns or relationships discovered
- What was the best result achieved

Keep it under 200 words. Be specific with numbers.

RUNS:
{json.dumps([{
    "run_id": r.run_id,
    "config": r.config.model_dump(),
    "eval_score": r.eval_score,
    "hypothesis": r.hypothesis,
    "status": r.status,
    "failure_reason": r.failure_reason,
} for r in runs], indent=2)}
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return self._fallback_compress(runs)

    def _format_runs_full(self, runs: list[RunResult]) -> str:
        lines = []
        for run in runs:
            score_str = f"{run.eval_score:.4f}" if run.eval_score else "N/A"
            status_str = run.status.upper()
            hypothesis = run.hypothesis[:200] + "..." if len(run.hypothesis) > 200 else run.hypothesis
            breakdown_str = json.dumps(run.eval_breakdown)
            if len(breakdown_str) > 300:
                breakdown_str = breakdown_str[:300] + "..."

            lines.append(
                f"[{run.run_id}] {status_str} | score={score_str}\n"
                f"  Hypothesis: {hypothesis}\n"
                f"  Config: lr={run.config.learning_rate}, "
                f"rank={run.config.lora_rank}, "
                f"epochs={run.config.epochs}, "
                f"scheduler={run.config.scheduler}\n"
                f"  Breakdown: {breakdown_str}\n"
                + (f"  Failed: {run.failure_reason}\n" if run.failure_reason else "")
            )

        return "\n".join(lines)

    def _fallback_compress(self, runs: list[RunResult]) -> str:
        """Simple string fallback if LLM compression fails."""
        scores = [r.eval_score for r in runs if r.eval_score]
        best = max(scores) if scores else 0
        avg = sum(scores) / len(scores) if scores else 0
        return (
            f"{len(runs)} earlier runs. "
            f"Best score: {best:.4f}. "
            f"Average score: {avg:.4f}. "
            f"Failed runs: {sum(1 for r in runs if r.status == 'failed')}."
        )