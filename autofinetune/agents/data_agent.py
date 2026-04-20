from loguru import logger
from autofinetune.agents.base import BaseAgent
from autofinetune.tools.schemas import DATA_TOOLS
from autofinetune.tools.data_cleaning import (
    deduplicate_dataset,
    filter_by_length,
    score_quality,
    fix_formatting,
)
from autofinetune.tools.format_conversion import (
    convert_to_chat_template,
    validate_schema,
    split_dataset,
)
from autofinetune.tools.benchmark_generation import (
    generate_benchmark_from_description,
    generate_benchmark_from_dataset,
    validate_benchmark_quality,
)
from autofinetune.tools.validation import (
    get_dataset_stats,
    check_leakage,
)


DATA_AGENT_PROMPT = """
You are a data preparation specialist for LLM finetuning.
Your job is to take a raw dataset and prepare it for training — 
cleaning it, converting it to the right format, and generating 
a benchmark to evaluate the finetuned model.

You have access to tools to:
- Clean and deduplicate datasets
- Filter by quality and length
- Convert to the correct chat template format for the base model
- Generate evaluation benchmarks tailored to the use case
- Validate dataset quality

Work methodically:
1. First get stats on the raw dataset to understand what you're working with
2. Clean and filter
3. Convert to the right format for the base model
4. Split into train/val
5. Generate or validate the benchmark
6. Return a quality report

Always call get_dataset_stats first before doing anything else.
Always validate the dataset schema after conversion.
"""


class DataAgent(BaseAgent):
    def __init__(self, model: str):
        super().__init__(
            model=model,
            system_prompt=DATA_AGENT_PROMPT,
            tools=DATA_TOOLS,
            temperature=0.2,        # low temp — this is methodical work
            max_iterations=15,      # may need many tool calls
        )

        self.tool_registry = {
            "deduplicate_dataset": deduplicate_dataset,
            "filter_by_length": filter_by_length,
            "score_quality": score_quality,
            "fix_formatting": fix_formatting,
            "convert_to_chat_template": convert_to_chat_template,
            "validate_schema": validate_schema,
            "split_dataset": split_dataset,
            "generate_benchmark_from_description": generate_benchmark_from_description,
            "generate_benchmark_from_dataset": generate_benchmark_from_dataset,
            "validate_benchmark_quality": validate_benchmark_quality,
            "get_dataset_stats": get_dataset_stats,
            "check_leakage": check_leakage,
        }

    def _build_prompt(self, task: str, context: dict) -> str:
        return f"""
TASK: {task}

DATASET PATH: {context['dataset_path']}
DATASET FORMAT: {context['dataset_format']}
BASE MODEL: {context['base_model']}
USE CASE: {context['use_case']}
BENCHMARK CONFIG: {context['benchmark_config']}

Prepare this dataset for finetuning. Generate a benchmark 
appropriate for evaluating: {context['use_case']}

Return when you have:
1. A cleaned, formatted dataset ready for training
2. A benchmark JSONL file for evaluation
3. A quality report summarizing what you found and did
"""

    def _parse_output(self, content: str) -> dict:
        """
        The data agent's final message should contain paths and a summary.
        We extract what we need or use sensible defaults.
        """
        import re

        processed_path = self._extract_path(content, "processed") or \
                         self._extract_path(content, "train") or \
                         self._extract_path(content, "cleaned")

        benchmark_path = self._extract_path(content, "benchmark") or \
                         self._extract_path(content, "eval")

        return {
            "processed_dataset_path": processed_path,
            "benchmark_path": benchmark_path,
            "quality_report": {
                "summary": content,
                "score": self._extract_quality_score(content),
            }
        }

    def _extract_path(self, content: str, keyword: str) -> str | None:
        pattern = rf"[\w./\-_]*{keyword}[\w./\-_]*\.jsonl"
        match = re.search(pattern, content, re.IGNORECASE)
        return match.group(0) if match else None

    def _extract_quality_score(self, content: str) -> float:
        import re
        match = re.search(r"quality[:\s]+(\d+\.?\d*)", content, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return score / 100 if score > 1 else score
        return 0.8  # default if not found