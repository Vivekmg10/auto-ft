DATA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_stats",
            "description": "Get statistics about a dataset — size, length distribution, format sample. Always call this first before doing anything else.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset JSONL file"
                    }
                },
                "required": ["dataset_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deduplicate_dataset",
            "description": "Remove duplicate and near-duplicate examples from a dataset using exact and fuzzy matching",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to input dataset JSONL"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save deduplicated dataset"
                    },
                    "fuzzy": {
                        "type": "boolean",
                        "description": "Also remove near-duplicates using MinHash. Slower but more thorough.",
                        "default": True
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Similarity threshold for fuzzy dedup 0-1. Default 0.85",
                        "default": 0.85
                    }
                },
                "required": ["dataset_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_length",
            "description": "Filter examples by token length. Removes too-short or too-long examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to input dataset JSONL"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save filtered dataset"
                    },
                    "min_tokens": {
                        "type": "integer",
                        "description": "Minimum token count to keep",
                        "default": 10
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum token count to keep",
                        "default": 2048
                    },
                    "tokenizer_name": {
                        "type": "string",
                        "description": "HuggingFace tokenizer to use for counting",
                        "default": "Qwen/Qwen2.5-7B-Instruct"
                    }
                },
                "required": ["dataset_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "score_quality",
            "description": "Score each example for quality using an LLM judge. Filters below threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to input dataset JSONL"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save quality-filtered dataset"
                    },
                    "use_case": {
                        "type": "string",
                        "description": "Description of the task — used to judge relevance"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum quality score 0-1 to keep. Default 0.6",
                        "default": 0.6
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Only score a sample for speed. -1 means score all.",
                        "default": -1
                    }
                },
                "required": ["dataset_path", "output_path", "use_case"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fix_formatting",
            "description": "Fix common formatting issues — broken JSON, encoding errors, whitespace, missing fields",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to input dataset JSONL"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save fixed dataset"
                    }
                },
                "required": ["dataset_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_to_chat_template",
            "description": "Convert dataset to the correct chat template format for the target base model",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to input dataset JSONL"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save converted dataset"
                    },
                    "base_model": {
                        "type": "string",
                        "description": "HuggingFace model name — determines which chat template to use"
                    },
                    "input_format": {
                        "type": "string",
                        "description": "Format of the input data",
                        "enum": ["sharegpt", "alpaca", "raw", "custom"],
                        "default": "sharegpt"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to inject into every conversation"
                    }
                },
                "required": ["dataset_path", "output_path", "base_model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_schema",
            "description": "Validate that a dataset is correctly formatted and ready for training. Returns a validation report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to dataset JSONL to validate"
                    },
                    "base_model": {
                        "type": "string",
                        "description": "Target base model — used to check template compatibility"
                    }
                },
                "required": ["dataset_path", "base_model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "split_dataset",
            "description": "Split dataset into train and validation sets",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to input dataset JSONL"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save train.jsonl and val.jsonl"
                    },
                    "val_ratio": {
                        "type": "number",
                        "description": "Fraction for validation set. Default 0.05",
                        "default": 0.05
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility",
                        "default": 42
                    }
                },
                "required": ["dataset_path", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_benchmark_from_description",
            "description": "Generate a benchmark eval set from a natural language description of the use case",
            "parameters": {
                "type": "object",
                "properties": {
                    "use_case": {
                        "type": "string",
                        "description": "Description of what the model should be good at"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save generated benchmark JSONL"
                    },
                    "n_examples": {
                        "type": "integer",
                        "description": "Number of benchmark examples to generate",
                        "default": 50
                    },
                    "difficulty_distribution": {
                        "type": "object",
                        "description": "Fraction of easy/medium/hard examples",
                        "default": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
                    }
                },
                "required": ["use_case", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_benchmark_from_dataset",
            "description": "Generate a benchmark by sampling and transforming examples from the training dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to training dataset JSONL"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save benchmark JSONL"
                    },
                    "n_examples": {
                        "type": "integer",
                        "description": "Number of benchmark examples to generate",
                        "default": 50
                    }
                },
                "required": ["dataset_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_benchmark_quality",
            "description": "Validate a benchmark for leakage against training data, ambiguity, and difficulty distribution",
            "parameters": {
                "type": "object",
                "properties": {
                    "benchmark_path": {
                        "type": "string",
                        "description": "Path to benchmark JSONL"
                    },
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to training dataset — used for leakage check"
                    }
                },
                "required": ["benchmark_path", "train_dataset_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_leakage",
            "description": "Check if benchmark examples overlap with training data",
            "parameters": {
                "type": "object",
                "properties": {
                    "benchmark_path": {
                        "type": "string",
                        "description": "Path to benchmark JSONL"
                    },
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to training dataset JSONL"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold to flag as leakage",
                        "default": 0.8
                    }
                },
                "required": ["benchmark_path", "train_dataset_path"]
            }
        }
    }
]   