import json
import re
from loguru import logger
from autofinetune.agents.base import BaseAgent


EVALUATOR_PROMPT = """
You are an expert LLM evaluator specializing in conversational AI quality.
Your job is to evaluate a finetuned model's checkpoint against a benchmark
and produce a detailed, honest assessment.

You evaluate across multiple dimensions:
- Task completion: Does the model actually solve the user's problem?
- Tone and empathy: Is the response appropriate for the context?
- Accuracy: Is the information correct and grounded?
- Consistency: Are similar questions answered consistently?
- Refusal quality: Does it gracefully handle out-of-scope requests?
- Conciseness: Is the response appropriately concise?

You use an LLM-as-judge approach with chain of thought scoring.
For each example: reason first, then score.

Your final output must be valid JSON with this structure:
{
  "primary_score": <float 0-1>,
  "breakdown": {
    "task_completion": <float 0-1>,
    "tone_empathy": <float 0-1>,
    "accuracy": <float 0-1>,
    "consistency": <float 0-1>,
    "refusal_quality": <float 0-1>,
    "conciseness": <float 0-1>
  },
  "strengths": [<string>, ...],
  "weaknesses": [<string>, ...],
  "recommendation": <string>
}
"""


class EvaluatorAgent(BaseAgent):
    def __init__(self, model: str):
        super().__init__(
            model=model,
            system_prompt=EVALUATOR_PROMPT,
            tools=[],
            temperature=0.1,
            max_iterations=1,
        )

    def _build_prompt(self, task: str, context: dict) -> str:
        benchmark_examples = self._load_benchmark(context["benchmark_path"])
        model_outputs = self._run_model_on_benchmark(
            checkpoint_path=context["checkpoint_path"],
            examples=benchmark_examples,
            base_model=context["base_model"],
        )

        formatted = self._format_examples_for_judge(
            benchmark_examples,
            model_outputs
        )

        return f"""
TASK: {task}

USE CASE: {context['use_case']}
PRIMARY METRIC: {context['primary_metric']}

BENCHMARK RESULTS ({len(benchmark_examples)} examples):
{formatted}

Evaluate the model's performance across all dimensions.
Be honest and specific. Cite examples for strengths and weaknesses.
Return your assessment as JSON.
"""

    def _parse_output(self, content: str) -> dict:
        try:
            pattern = r"```json\s*(.*?)```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # try parsing raw JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))

        except Exception as e:
            logger.error(f"Failed to parse evaluator output: {e}")

        return {
            "primary_score": 0.0,
            "breakdown": {},
            "strengths": [],
            "weaknesses": ["Parse error — could not extract scores"],
            "recommendation": "Manual review required"
        }

    def _load_benchmark(self, benchmark_path: str) -> list[dict]:
        import jsonlines
        examples = []
        with jsonlines.open(benchmark_path) as reader:
            for item in reader:
                examples.append(item)
        return examples[:50]  # cap at 50 examples per eval to control cost

    def _run_model_on_benchmark(
        self,
        checkpoint_path: str,
        examples: list[dict],
        base_model: str,
    ) -> list[str]:
        """
        Runs the finetuned checkpoint on benchmark prompts.
        Returns model outputs as a list of strings.
        """
        from autofinetune.eval.harness import run_checkpoint_on_examples

        return run_checkpoint_on_examples(
            checkpoint_path=checkpoint_path,
            examples=examples,
            base_model=base_model,
        )

    def _format_examples_for_judge(
        self,
        examples: list[dict],
        outputs: list[str],
    ) -> str:
        formatted = []
        for i, (example, output) in enumerate(zip(examples, outputs)):
            formatted.append(
                f"Example {i+1}:\n"
                f"Input: {example.get('input', example.get('prompt', ''))}\n"
                f"Expected: {example.get('output', example.get('ideal', ''))}\n"
                f"Model output: {output}\n"
                f"Rubric: {example.get('rubric', 'General quality')}\n"
                f"---"
            )
        return "\n".join(formatted)