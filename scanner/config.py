"""
Configuration dataclasses for the vulnerability scanner pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScannerConfig:
    # Module 1: Detection
    model_path: str = ""
    faiss_dir: str = ""
    tokenizer_name: str = "microsoft/graphcodebert-base"
    model_name_or_path: str = "microsoft/graphcodebert-base"
    block_size: int = 400
    hidden_size: int = 128
    feature_dim_size: int = 768
    num_GNN_layers: int = 2
    gnn: str = "ReGCN"
    format: str = "uni"
    window_size: int = 5
    remove_residual: bool = False
    att_op: str = "mul"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    use_faiss: bool = True
    embed_dim: int = 512
    num_classes: int = 2

    # Hybrid scoring
    beta: float = 0.6
    top_k: int = 5
    detection_threshold: float = 0.5

    # Module 2: Exploit retrieval
    exploit_db_dir: str = ""
    exploit_top_k: int = 3

    # Module 3: LLM adaptation
    llm_backend: Optional[str] = None
    llm_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    llm_api_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_max_retries: int = 3
    llm_temperature: float = 0.0
    llm_cache_results: bool = True

    # Module 4: Sandbox
    sandbox_memory: str = "1g"
    sandbox_cpu: float = 0.9
    sandbox_timeout: int = 30
    sandbox_network: str = "none"
    sandbox_docker_image: str = ""
    sandbox_use_valgrind: bool = False
    sandbox_use_seccomp: bool = True
    sandbox_use_multi_sanitizer: bool = False
    sandbox_use_libfuzzer: bool = False
    sandbox_fuzz_time: int = 15
    sandbox_prefer_clang: bool = False

    # Taint analysis
    use_taint_analysis: bool = False
    joern_path: Optional[str] = None

    # Triage Head (LLM-based verification)
    use_triage: bool = False
    triage_model: str = "openrouter/anthropic/claude-sonnet-4-6"
    triage_temperature: float = 0.1
    triage_max_tokens: int = 2048
    triage_api_key: Optional[str] = None

    # Pipeline
    max_exploits_per_sample: int = 3
    early_stop_on_confirmed: bool = True
    language: str = "c"
    use_multi_harness: bool = True

    # Dry-run mode (skips sandbox + triage API, uses stubs)
    dry_run: bool = False

    # Hardware
    device: str = "cpu"
    no_cuda: bool = False
