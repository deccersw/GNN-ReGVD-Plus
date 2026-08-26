# GNN-ReGVD+

**A vulnerability detection pipeline for C/C++ code combining GNN-based detection with execution-based verification.**

GNN-ReGVD+ extends [ReGVD](https://arxiv.org/abs/2110.07317) (ICSE 2022) into a full vulnerability scanner: neural network detection is followed by automatic exploit generation and sandbox verification, so no verdict is issued without execution-based evidence.

---

## Key Results

| Metric | ReGVD baseline | GNN-ReGVD+ (Hybrid) | + LoRA (domain transfer) |
|--------|---------------|---------------------|--------------------------|
| Accuracy | 0.646 | 0.645 | **0.928** |
| ROC-AUC | 0.704 | 0.700 | **0.751** (+0.179 on new domain) |
| F1 | 0.571 | 0.575 | **0.284** |
| Precision | 0.603 | 0.599 | **0.367** (5x improvement) |
| Trainable params | ~12.7M | ~13.0M | **~152.8K** (0.12%) |
| VRAM (fine-tuning) | ~14.2 GB | — | **~7.8 GB** (1.8x reduction) |
| Energy (fine-tuning) | ~1.92 kWh | — | **~0.23 kWh** (8x reduction) |
| Inference latency | ~28 ms | ~34 ms | — |

*Evaluated on Devign (27,318 functions) and MegaVul (339,548 functions). Hardware: NVIDIA RTX 3090.*

**Sandbox verification:** 18 vulnerability types, 23 CWE mappings, 71 detection patterns.
**LLM triage head:** 81.6% recall on vulnerable class, $0.006/query (Claude Sonnet).
**Codebase:** 9,372 lines of Python, 203 automated tests.

---

## Architecture

```
Source Code (C/C++)
        |
  [Module 1] GNN Detection
  GraphCodeBERT (frozen, 125M) + LoRA (r=8, a=16)
  ReGCN (2 layers) + Soft Attention Pooling
  -> Classification Head (P_cls)
  -> Embedding Head (512-d) + FAISS k-NN (S_faiss, vuln type)
  -> S_hybrid = beta * P_cls + (1-beta) * S_faiss
        |
  [Module 2] Exploit Retrieval
  FAISS index, 20 templates across 10 categories
        |
  [Module 3] LLM Adaptation (cascade)
  Level 1: LLM (Qwen2.5-Coder / OpenAI API) + iterative refinement
  Level 2: Template substitution (fallback)
  Level 3: Generic harness generators (guaranteed output)
        |
  [Module 4] Sandbox Verification
  Docker (seccomp, no-net, cap-drop=ALL)
  Sanitizers: ASan+UBSan | MSan | TSan
  Coverage-guided fuzzing: LibFuzzer
  Evidence classification (71 patterns -> 4 levels)
        |
  Verdict: CONFIRMED / SUGGESTIVE / NEUTRAL / SAFE
  + CWE ID + confidence score + exploit trace

  [Parallel] LLM Triage Head
  Claude Sonnet + CodeQL graph context -> cross-validation
```

---

## Vulnerability Coverage

| Type | CWE | Templates | Sanitizer | Taint |
|------|-----|-----------|-----------|-------|
| Buffer Overflow (stack) | CWE-121 | 3 | ASan | + |
| Buffer Overflow (heap) | CWE-122 | 2 | ASan | + |
| Use-After-Free | CWE-416 | 2 | ASan | + |
| Double-Free | CWE-415 | -- | ASan | + |
| Format String | CWE-134 | 2 | ASan | + |
| Integer Overflow | CWE-190 | 2 | UBSan | |
| Null Pointer Deref | CWE-476 | 2 | ASan | |
| Off-by-One | CWE-193 | 2 | ASan | + |
| Division by Zero | CWE-369 | 2 | UBSan | |
| Uninitialized Read | CWE-457 | 2 | MSan | |
| Data Race | CWE-362 | 1 | TSan | |
| Deadlock | CWE-833 | -- | TSan | |

**Total:** 18 types, 20 templates, 15 generic generators, 23 CWE mappings.

---

## Installation

```bash
git clone https://github.com/deccersw/GNN-ReGVD-Plus.git
cd GNN-ReGVD

python -m venv .venv
source .venv/bin/activate
pip install torch>=2.0.0 transformers>=4.30.0 numpy>=1.24.0 faiss-cpu>=1.7.4 pytest>=7.0.0

# (Optional) Docker images for sandbox verification
docker build -t gnn-regvd-sandbox:c-cpp -f sandbox/dockerfiles/Dockerfile.c_cpp sandbox/dockerfiles/
docker build -t gnn-regvd-sandbox:clang -f sandbox/dockerfiles/Dockerfile.clang sandbox/dockerfiles/

# (Optional) LLM adapter
pip install accelerate>=0.20.0 sentencepiece>=0.1.99
```

System dependencies: `gcc`, `clang` (for local compilation). Docker (for sandbox). Joern (optional, for interprocedural taint analysis).

---

## Usage

### Sandbox-only mode (no trained model required)

```bash
# Single function
python scan_cli.py --sandbox-only \
    --code "void f(const char *s){char b[8];strcpy(b,s);}" --verbose

# From file
python scan_cli.py --sandbox-only --file vuln.c --verbose

# With LLM adaptation (via Ollama)
python scan_cli.py --sandbox-only \
    --code "void f(char *s){char b[8];strcpy(b,s);}" \
    --llm-backend api \
    --llm-api-url "http://localhost:11434/v1/chat/completions" \
    --llm-model "qwen2.5-coder:7b"
```

### Project-level scanning (Module 0: interprocedural inlining)

The scanner analyses functions, but a vulnerability is often only visible when a
caller and its callee are read together — and they usually live in different
files. Module 0 splits a whole project into *analysis units*: each function with
the bodies of the functions it calls inlined into it, up to a configurable
depth and within the detector's token window.

```bash
# Split the project into units and stop — no GNN, no LLM, no sandbox.
# Use this to inspect the split on its own.
python scan_cli.py --project ./myproject --build-units-only \
    --inline-depth 2 --units-out units.jsonl --units-stats stats.json
```

```bash
# Same thing through the module's own CLI, with the units printed out
python -m interproc.cli --project ./myproject --inline-depth 2 --show 3
```

```bash
# Full pipeline over a project
python scan_cli.py --project ./myproject --inline-depth 2 \
    --model-path code/saved_models/lora_faiss/checkpoint-best-acc/model.bin \
    --faiss-dir code/saved_models/lora_faiss/faiss_index
```

`--inline-depth` is the depth hyperparameter: `0` reproduces the current
single-function behaviour byte for byte, `1` inlines direct callees, `2` also
inlines theirs. Because the detector only sees `block_size - 2 = 398` BPE
tokens, call sites are inlined in order of security relevance (callees
containing dangerous sinks and callees receiving caller-derived arguments go
first) rather than in source order; `--inline-strategy` switches that off.

Each unit carries two views of the code: an inlined one for the GNN, and a
self-contained compilable bundle for the LLM harness generator and the sandbox.

### Full pipeline (GNN + Exploit + Sandbox)

```bash
# Train the model
cd code
python run.py --do_train --do_eval --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --output_dir=./saved_models/lora_faiss \
    --use_lora --use_faiss --embed_dim 512 \
    --gnn ReGCN --epoch 100 --learning_rate 5e-4

# Scan
cd ..
python scan_cli.py --file vuln.c \
    --model-path code/saved_models/lora_faiss/checkpoint-best-acc/model.bin \
    --faiss-dir code/saved_models/lora_faiss/faiss_index \
    --multi-sanitizer --use-libfuzzer --taint-analysis --verbose
```

### Batch scanning

```bash
python scan_cli.py --jsonl dataset/test.jsonl --limit 50 --output results.json --json
python evaluate.py --mode sandbox-only --limit 20 --timeout 10
```

---

## Project Structure

```
GNN-ReGVD/
├── interproc/              # Module 0: Interprocedural Inlining
│   ├── clex.py             #   C/C++ lexical scanner (code vs string/comment)
│   ├── discovery.py        #   Project walk + parse cache
│   ├── callgraph.py        #   Call resolution (TU rules, arity, SCC)
│   ├── inliner.py          #   Depth-limited, budget-aware expansion
│   ├── bundler.py          #   Compilable dependency bundle for the sandbox
│   └── cli.py              #   Standalone CLI (python -m interproc.cli)
├── code/                   # Module 1: GNN Detection
│   ├── model.py            #   GNNReGVD + LoRA + EmbeddingHead
│   ├── modelGNN_updates.py #   ReGCN, ReGGNN with residual connections
│   ├── faiss_index.py      #   FAISS index manager
│   ├── inference.py        #   Hybrid inference + beta tuning
│   ├── losses.py           #   SupCon + Triplet losses
│   ├── run.py              #   Training pipeline
│   └── run_finetune.py     #   Hot update + LoRA fine-tune
├── exploit_db/             # Module 2-3: Exploit Retrieval + Adaptation
│   ├── exploit_index.py    #   FAISS index for exploit templates
│   ├── exploit_adapter.py  #   LLM / template / generic cascade
│   └── templates/          #   20 exploit templates (10 categories)
├── sandbox/                # Module 4: Sandbox Verification
│   ├── executor.py         #   Docker/local + multi-sanitizer + LibFuzzer
│   ├── evidence.py         #   Evidence classification + CWE mapping
│   └── dockerfiles/        #   Dockerfile.c_cpp, Dockerfile.clang, seccomp
├── analysis/               # Taint Analysis
│   └── taint.py            #   Joern CPG / regex heuristic
├── scanner/                # Pipeline Orchestrator
│   ├── pipeline.py         #   VulnerabilityScanner (lazy init)
│   ├── config.py           #   ScannerConfig (36 params)
│   └── report.py           #   Text/JSON reports
├── triage/                 # LLM Triage Head
│   ├── triage_head.py      #   Claude Sonnet + CodeQL integration
│   └── prompts_c.py        #   Structured prompts for C/C++
├── tests/                  # 203 automated tests (~6s, no GPU/Docker needed)
├── test_samples/           # Sample vulnerable/safe C files
├── dataset/                # Devign train/valid/test splits (JSONL)
├── article/                # Paper source (LNCS format, RU + EN)
├── scan_cli.py             # CLI interface
└── evaluate.py             # Evaluation metrics
```

---

## Testing

203 tests, ~6 seconds. No GPU, Docker, or trained model required.

```bash
python -m pytest tests/ -v --ignore=tests/test_faiss_vuln_type.py
```

| Test file | Tests | Coverage |
|-----------|-------|----------|
| test_sandbox.py | 61 | Evidence classification, ASan, Valgrind, Docker, batch |
| test_enhancements.py | 55 | MSan, TSan, CWE mapping, LibFuzzer, taint, templates |
| test_adapter.py | 36 | LLM retry, JSON parsing, multi-harness, cache |
| test_pipeline.py | 25 | Orchestration, verdict, confidence, CLI, e2e |
| test_evaluate.py | 18 | Metrics, dataset loading, confusion matrix |

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lora_rank` | 8 | Balance between capacity and parameter count |
| `lora_alpha` | 16 | Scaling = alpha/rank = 2.0 (LoRA standard) |
| `embed_dim` | 512 | Sufficient for code pattern discrimination |
| `contrastive_weight` | 0.3 | 70% BCE + 30% SupCon |
| `temperature` | 0.07 | Standard for supervised contrastive loss |
| `beta` | 0.6 | Classifier-dominant hybrid (tuned via grid search) |
| `top_k` | 5 | FAISS neighbors for scoring |
| `replay_ratio` | 0.2 | Prevents catastrophic forgetting during LoRA fine-tuning |

---

## Data Format

**Input (JSONL):**
```json
{"func": "int vuln(char *buf) { strcpy(dest, buf); return 0; }", "target": 1, "idx": 0}
```

**Output (ScanResult):**
```json
{
  "verdict": "CONFIRMED",
  "confidence": 0.95,
  "vuln_type": "buffer_overflow",
  "cwe_id": "CWE-122",
  "exploits_tried": 2,
  "exploits_confirmed": 1
}
```

---

## Citation

```bibtex
@inproceedings{antonov2025gnnregvdplus,
  author    = {Antonov, A.V. and Burovin, V.S. and Voevodkin, V.S.},
  title     = {{GNN-ReGVD+}: A Vulnerability Detection Pipeline for {C/C++}
               Code with Execution-Based Verification},
  booktitle = {HSE University Coursework},
  year      = {2025}
}
```

Original ReGVD paper:
```bibtex
@inproceedings{NguyenReGVD,
  author    = {Van-Anh Nguyen and Dai Quoc Nguyen and Van Nguyen and Trung Le
               and Quan Hung Tran and Dinh Phung},
  title     = {{ReGVD}: Revisiting Graph Neural Networks for Vulnerability Detection},
  booktitle = {ICSE '22 Companion},
  year      = {2022}
}
```

## License

This project is distributed on an "AS IS" basis, without warranties or conditions of any kind.
