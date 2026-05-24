# GNN-ReGVD: Техническая спецификация

## Полная документация интегрированного сканера уязвимостей

---

## 1. Обзор системы

GNN-ReGVD — интегрированный сканер уязвимостей в исходном коде C/C++, объединяющий нейросетевую детекцию (GNN), поиск эксплойтов по FAISS-индексу, LLM-адаптацию эксплойтов, изолированную верификацию в Docker-песочнице с мульти-санитайзерной поддержкой (ASan+UBSan, MSan, TSan), покрытие-ориентированным фаззингом (LibFuzzer) и статическим taint-анализом (Joern/эвристический).

Система синтезирует подходы двух академических работ:
- **"Verify Before You Fix"** (Gajjar, 2026) — execution-based валидация через Docker sandbox с evidence classification
- **"SAST-Genius"** (Agrawal & Ahi, 2025) — LLM-triage и PoC-генерация для SAST-алертов

### 1.1. Ключевой принцип

Никакой вердикт о уязвимости не выносится без execution-based подтверждения. GNN-детекция выявляет кандидатов, а sandbox либо подтверждает эксплуатируемость (CONFIRMED), либо фильтрует ложные срабатывания (FP reduction).

### 1.2. Архитектура пайплайна

```
SOURCE CODE (C/C++)
     |
     v
+--------------------------------------+
| STEP 0: Taint Analysis (опционально) |  Joern CPG / regex-эвристика
| source → sink data flow paths        |  → taint_context для LLM
+--------------------------------------+
     |
     v
+--------------------------------------+
| MODULE 1: SAST Detection             |  GraphCodeBERT + LoRA → GNN (ReGCN)
| hybrid_score = β·cls + (1-β)·faiss   |  → Hybrid Score (classifier + FAISS)
|                                       |  → vuln_type via k-NN majority vote
+--------------------------------------+
     | {code, vuln_type, confidence, embedding}
     v
+--------------------------------------+
| MODULE 2: Exploit Retrieval           |  Отдельный FAISS-индекс для эксплойтов
|                                       |  Двухступенчатый поиск: similarity + type
+--------------------------------------+
     | [{payload, template, similarity}]
     v
+--------------------------------------+
| MODULE 3: Exploit Adaptation          |  LLM (Qwen2.5-Coder-7B / API)
|                                       |  Iterative refinement (до 3 попыток)
|                                       |  Taint context → улучшенные exploit'ы
|                                       |  Fallback: template → generic harness
+--------------------------------------+
     | [{harness_code, payload, expected_signals}]
     v
+--------------------------------------+
| MODULE 4: Sandbox Verification        |  Docker/Local + Multi-Sanitizer:
|   ┌─ ASan+UBSan (gcc/clang)          |    - buffer overflow, UAF, double-free
|   ├─ MSan (clang only)               |    - uninitialized memory read
|   └─ TSan (clang only)               |    - data race, deadlock
|                                       |  + LibFuzzer (coverage-guided fuzzing)
|                                       |  + Valgrind (опционально)
|                                       |  + seccomp, cap-drop, network=none
|                                       |  Evidence: CONFIRMED/SUGGESTIVE/NEUTRAL
+--------------------------------------+
     |
     v
  FINAL VERDICT + CWE ID + exploit trace + confidence
```

---

## 2. Структура проекта

```
GNN-ReGVD/
├── code/                              # Модуль 1: GNN-детекция
│   ├── model.py                       # GNNReGVD: GraphCodeBERT + LoRA + EmbeddingHead
│   ├── modelGNN_updates.py            # ReGCN, ReGGNN, GGGNN, GraphConvolution
│   ├── faiss_index.py                 # FAISSIndexManager + compute_vuln_type()
│   ├── inference.py                   # HybridPredictor: dual scoring pipeline
│   ├── losses.py                      # SupervisedContrastiveLoss, TripletMarginLoss
│   ├── run.py                         # Training pipeline: TextDataset, training loop
│   ├── run_finetune.py                # Hot update / LoRA fine-tune
│   └── utils.py                       # Preprocessing: adj, features
│
├── exploit_db/                        # Модуль 2+3: Exploit Retrieval + Adaptation
│   ├── __init__.py
│   ├── exploit_index.py               # ExploitDBManager: FAISS-индекс эксплойтов
│   ├── exploit_adapter.py             # ExploitAdapter: LLM/template/generic (909 строк)
│   ├── exploit_loader.py              # Загрузчик шаблонов и JSONL
│   └── templates/                     # 20 JSON-шаблонов эксплойтов по 10 категориям
│       ├── buffer_overflow.json       # 3 шаблона (strcpy, sprintf, memcpy)
│       ├── heap_buffer_overflow.json  # 2 шаблона (malloc, realloc)
│       ├── use_after_free.json        # 2 шаблона (basic UAF, double-free)
│       ├── format_string.json         # 2 шаблона (read, write)
│       ├── integer_overflow.json      # 2 шаблона (signed, alloc)
│       ├── null_pointer_deref.json    # 2 шаблона (basic, return-NULL)
│       ├── off_by_one.json            # 2 шаблона (loop, strlen)
│       ├── division_by_zero.json      # 2 шаблона (basic, modulo)
│       ├── uninitialized_read.json    # 2 шаблона (heap, stack)
│       └── race_condition.json        # 1 шаблон (data race)
│
├── sandbox/                           # Модуль 4: Sandbox Verification
│   ├── __init__.py
│   ├── executor.py                    # SandboxExecutor: Docker/local + multi-sanitizer + LibFuzzer (650 строк)
│   ├── evidence.py                    # EvidenceClassifier: ASan/UBSan/MSan/TSan/Valgrind + CWE (439 строк)
│   ├── docker_manager.py              # Docker image build/cleanup
│   └── dockerfiles/
│       ├── Dockerfile.c_cpp           # gcc:14 + valgrind + sandbox user
│       ├── Dockerfile.clang           # clang:18 для MSan/TSan/LibFuzzer
│       └── seccomp_sandbox.json       # Seccomp-профиль: whitelist syscalls
│
├── analysis/                          # Taint Analysis (Step 0)
│   ├── __init__.py
│   └── taint.py                       # TaintAnalyzer: Joern CPG / regex-эвристика (402 строки)
│
├── scanner/                           # Оркестратор пайплайна
│   ├── __init__.py
│   ├── pipeline.py                    # VulnerabilityScanner: end-to-end (444 строки)
│   ├── config.py                      # ScannerConfig: все параметры (75 строк)
│   └── report.py                      # Генерация отчётов: text, JSON, batch summary
│
├── tests/                             # 203 теста
│   ├── test_sandbox.py                # 61 тест: evidence, confidence, ASan, Valgrind, docker
│   ├── test_adapter.py                # 36 тестов: LLM retry, JSON parsing, multi-harness, cache
│   ├── test_pipeline.py               # 25 тестов: orchestration, verdict, confidence, CLI, e2e
│   ├── test_evaluate.py               # 18 тестов: metrics, dataset loading, confusion matrix
│   ├── test_enhancements.py           # 55 тестов: MSan, TSan, CWE, LibFuzzer, taint analysis
│   ├── test_exploit_db.py             # 7 тестов: ExploitDBManager (требует faiss)
│   └── test_faiss_vuln_type.py        # 4 теста: compute_vuln_type (требует faiss)
│
├── dataset/
│   ├── train.jsonl                    # Тренировочный набор: 21 854 сэмпла
│   ├── valid.jsonl                    # Валидационный набор: 2 732 сэмпла
│   └── test.jsonl                     # Тестовый набор: 2 732 сэмпла (1477 safe, 1255 vuln)
│
├── scan_cli.py                        # CLI-интерфейс сканера
├── evaluate.py                        # Скрипт оценки: метрики, бенчмарк
├── TECHNICAL_SPEC.md                  # Техническая спецификация (этот файл)
└── PLAN.md                            # Архитектурный план проекта
```

**Общий объём кода:** ~9 372 строки Python.

---

## 3. Полный процесс работы пайплайна

### 3.1. Обзор жизненного цикла запроса

При подаче исходного кода C/C++ на вход сканер выполняет последовательность шагов. Каждый шаг принимает решение о дальнейшем ходе анализа на основе результатов предыдущего.

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Принимается исходный код (строка, файл или JSONL-пакет)  │
│ 2. [Опц.] Taint analysis: находятся пути source → sink      │
│ 3. GNN-детекция: вычисляется hybrid_score                    │
│    └─ Если < threshold → SAFE, pipeline завершается          │
│ 4. Определяется vuln_type через k-NN голосование в FAISS     │
│ 5. По эмбеддингу ищутся похожие exploit-шаблоны              │
│ 6. LLM адаптирует exploit'ы под конкретный код               │
│    └─ С учётом taint контекста, если доступен                │
│ 7. Sandbox верифицирует каждый harness:                      │
│    ├─ ASan+UBSan (обязательно)                               │
│    ├─ [Опц.] MSan (неинициализированная память)              │
│    ├─ [Опц.] TSan (гонки данных)                             │
│    ├─ [Опц.] LibFuzzer (если статический payload не сработал)│
│    └─ [Опц.] Valgrind (дополнительная проверка)              │
│ 8. Агрегация вердикта: CONFIRMED / SUGGESTIVE / NEUTRAL     │
│ 9. Генерация отчёта с CWE ID и evidence trace               │
└──────────────────────────────────────────────────────────────┘
```

### 3.2. Шаг 0: Taint Analysis (опциональный)

Перед нейросетевой детекцией система может провести статический taint-анализ кода для извлечения путей потока данных от источников (user input) до опасных стоков (dangerous sinks).

**Два бэкенда:**

| Бэкенд | Точность | Требования | Время |
|--------|----------|------------|-------|
| Joern CPG | Высокая (inter-procedural) | Joern CLI | 5–30с |
| Regex-эвристика | Средняя (intra-procedural) | Нет | <1с |

**Источники (INPUT_SOURCES, 18 шт.):**
`argv`, `argc`, `getenv`, `gets`, `fgets`, `scanf`, `fscanf`, `sscanf`, `read`, `recv`, `recvfrom`, `recvmsg`, `fread`, `getchar`, `getc`, `fgetc`, `getline`, `stdin`

**Опасные стоки (DANGEROUS_SINKS, 5 категорий):**

| Категория | Функции |
|-----------|---------|
| memory | strcpy, strcat, sprintf, vsprintf, gets, memcpy, memmove, bcopy |
| format | printf, fprintf, sprintf, snprintf, syslog, vprintf, vfprintf, vsprintf, vsnprintf |
| exec | system, popen, exec, execl, execle, execlp, execv, execve, execvp, execvpe |
| file | fopen, open, freopen, tmpnam, mktemp |
| alloc | malloc, calloc, realloc, alloca, free |

**Выход taint-анализа** передаётся в Модуль 3 как контекст для LLM, позволяя генерировать более точные exploit'ы, нацеленные на конкретные пути данных:

```python
TaintResult:
  taint_paths: [TaintPath(source="gets", line=5, sink="strcpy", line=8, category="memory")]
  call_graph:  [CallEdge(caller="main", callee="vulnerable_func", line=10)]
  functions:   [FunctionInfo(name="vulnerable_func", start=3, end=12)]
```

### 3.3. Шаг 1: GNN-детекция (Module 1)

Модель **GNNReGVD** обрабатывает исходный код через два параллельных пути:

```
Исходный код (до 400 токенов)
  │
  ├─> GraphCodeBERT (frozen 125M params)
  │   └─> LoRA-адаптеры (rank=8, alpha=16, ~1M params)
  │       └─> CLS-токен → cls_prob ∈ [0, 1]
  │
  ├─> Граф кода (AST + sliding window → adj matrix)
  │   └─> GNN (ReGCN, 2 слоя с residual connections)
  │       └─> Graph pooling → graph features
  │
  └─> EmbeddingHead (Linear 768→512, L2-norm)
      └─> embedding ∈ R^512 для FAISS
```

**Hybrid scoring:**
```
hybrid_score = β × cls_prob + (1 - β) × faiss_score
```

Где `faiss_score` — доля уязвимых среди top-K ближайших соседей в FAISS-индексе.

**Определение типа уязвимости** выполняется через majority vote по полю `vuln_type` у k ближайших уязвимых соседей.

Если `hybrid_score < detection_threshold` (по умолчанию 0.5) — возвращается вердикт **SAFE**, и пайплайн завершается.

### 3.4. Шаг 2: Exploit Retrieval (Module 2)

512-мерный эмбеддинг из Модуля 1 используется для поиска в отдельном FAISS-индексе exploit-шаблонов.

**Двухступенчатый поиск:**
1. FAISS similarity search (broad, top_k × 5)
2. Фильтрация по `vuln_type`, возврат top_k (по умолчанию 3)

**20 шаблонов в 10 категориях:**

| Категория | Шаблонов | Покрытие |
|-----------|----------|----------|
| buffer_overflow | 3 | strcpy, sprintf, memcpy overflow |
| heap_buffer_overflow | 2 | malloc overwrite, realloc overflow |
| use_after_free | 2 | basic UAF, double-free |
| format_string | 2 | read (%x), write (%n) |
| integer_overflow | 2 | signed overflow, alloc overflow |
| null_pointer_deref | 2 | basic NULL, return-NULL |
| off_by_one | 2 | loop boundary, strlen |
| division_by_zero | 2 | basic div/0, modulo |
| uninitialized_read | 2 | heap alloc, stack alloc |
| race_condition | 1 | data race без синхронизации |

### 3.5. Шаг 3: Exploit Adaptation (Module 3)

Найденные шаблоны адаптируются под конкретный уязвимый код с помощью цепочки фоллбэков:

```
1. LLM адаптация (Qwen2.5-Coder-7B или API)
   │ Iterative refinement: до 3 попыток с feedback
   │ Taint context → LLM знает конкретные source→sink пути
   │ Ошибка компиляции → включается в следующий промпт
   │
   ├── Успех → ExploitHarness (с валидацией синтаксиса)
   └── Неудача ↓

2. Template-based адаптация
   │ Подстановка {vulnerable_code} и {func_name}
   │
   ├── Шаблон найден → ExploitHarness
   └── Шаблон пуст ↓

3. Generic harness по vuln_type (14 генераторов + fallback)
   └── Всегда возвращает валидный harness
```

**14 специализированных vuln-type hints** направляют LLM:

| Тип | Стратегия exploit'а | Sanitizer |
|-----|---------------------|-----------|
| buffer_overflow | memset 4096 'A' + вызов | ASan |
| stack_buffer_overflow | то же | ASan |
| heap_buffer_overflow | malloc(32) + memset(256) | ASan |
| use_after_free | malloc + free + dereference | ASan |
| double_free | malloc + free + free | ASan |
| integer_overflow | INT_MAX + 1 | UBSan |
| format_string | "%x.%x.%x.%n" | ASan |
| null_pointer_deref | func(NULL) | ASan |
| off_by_one | buffer[size] вместо buffer[size-1] | ASan |
| division_by_zero | divisor = 0 | UBSan / SIGFPE |
| uninitialized_read | malloc без memset → read | MSan (clang) |
| race_condition | pthreads без mutex | TSan (clang) |
| stack_exhaustion | setrlimit + глубокая рекурсия | SIGSEGV |
| alloc_dealloc_mismatch | malloc + delete / new + free | ASan |

**Multi-harness генерация (Asymmetric Validation):**
Метод `adapt_multi()` генерирует несколько вариантов harness для одной уязвимости, дедуплицируя по MD5-хешу кода. Это увеличивает вероятность обнаружения конкретного способа триггеринга.

### 3.6. Шаг 4: Sandbox Verification (Module 4)

Каждый harness выполняется в изолированной среде. Система поддерживает два режима: **single-sanitizer** (по умолчанию) и **multi-sanitizer**.

#### 3.6.1. Single-Sanitizer Mode

Стандартный режим: компиляция с `-fsanitize=address,undefined`, запуск, классификация evidence.

```
harness.c → gcc -fsanitize=address,undefined -g → ./harness < payload
                                                      │
                                                      ├─ ASan/UBSan output → CONFIRMED
                                                      ├─ Signal (SEGV/ABRT) → CONFIRMED
                                                      ├─ Non-zero exit     → SUGGESTIVE
                                                      └─ Clean exit        → SAFE
```

#### 3.6.2. Multi-Sanitizer Mode (`--multi-sanitizer`)

Три профиля санитайзеров, взаимоисключающие (нельзя комбинировать ASan+MSan или ASan+TSan):

| Профиль | Компилятор | Флаги | Детектирует |
|---------|------------|-------|-------------|
| **asan_ubsan** | gcc / clang | `-fsanitize=address,undefined -g -fno-omit-frame-pointer` | buffer overflow, UAF, double-free, null deref, format string, integer overflow |
| **msan** | clang only | `-fsanitize=memory -g -fno-omit-frame-pointer -fno-optimize-sibling-calls` | uninitialized memory read (CWE-457) |
| **tsan** | clang only | `-fsanitize=thread -g -lpthread` | data race (CWE-362), deadlock (CWE-833), thread leak (CWE-404) |

**Интеллектуальный выбор профилей** (`_select_profiles`):
- По умолчанию всегда включается `asan_ubsan`
- Если `vuln_type` ∈ {uninitialized_read, uninitialized_variable} → добавляется `msan`
- Если `vuln_type` ∈ {race_condition, deadlock, toctou} → добавляется `tsan`
- Для каждого профиля компилируется и запускается отдельный бинарник

#### 3.6.3. LibFuzzer Mode (`--use-libfuzzer`)

Двухфазная стратегия вместо только статических payload:

```
Фаза 1: Статический payload (timeout 1с)
   │
   ├── CONFIRMED → завершение (быстрый путь)
   └── Не подтверждено ↓

Фаза 2: LibFuzzer (coverage-guided, timeout fuzz_time сек)
   │ Автотрансформация: main() → LLVMFuzzerTestOneInput()
   │ Компиляция: clang -fsanitize=fuzzer,address,undefined
   │ Запуск: ./harness_fuzz -max_total_time=15
   │
   └── LibFuzzer автоматически мутирует входы для покрытия
       новых путей выполнения
```

**Трансформация harness для LibFuzzer:**
Система автоматически преобразует harness с `main()` в формат LibFuzzer:
1. Извлекается имя вызываемой функции из main()
2. Заменяется main() на `LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)`
3. Добавляются необходимые include'ы

#### 3.6.4. Docker-изоляция

**Два Docker-образа:**

| Образ | Базовый | Назначение |
|-------|---------|------------|
| `Dockerfile.c_cpp` | gcc:14 | ASan+UBSan, Valgrind |
| `Dockerfile.clang` | silkeh/clang:18 | MSan, TSan, LibFuzzer |

**Параметры контейнера:**

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `--memory` | 1g | Лимит RAM |
| `--cpus` | 0.9 | Лимит CPU |
| `--network` | none | Полная сетевая изоляция |
| `--pids-limit` | 128 | Защита от fork bomb |
| `--security-opt` | no-new-privileges | Нет повышения привилегий |
| `--cap-drop` | ALL | Сброс всех capabilities |
| `--read-only` | (root fs) | Только чтение корневой FS |
| `--tmpfs /tmp` | size=256m, noexec, nosuid | Временные файлы |
| `--tmpfs /out` | size=64m, exec | Скомпилированный бинарник |

**Seccomp-профиль** (`seccomp_sandbox.json`):
- Default action: `SCMP_ACT_ERRNO` (блокировка по умолчанию)
- Whitelist: 70+ системных вызовов (read, write, mmap, clone, execve, etc.)
- Заблокированы: socket(), connect(), bind(), ptrace(), mount(), reboot()

#### 3.6.5. Evidence Classification

Четырёхуровневая классификация с confidence scoring.

**Паттерны детекции (6 категорий):**

| Категория | Количество паттернов | Примеры |
|-----------|---------------------|---------|
| ASan | 17 | heap-buffer-overflow, use-after-free, double-free, container-overflow |
| UBSan | 14 | signed integer overflow, division by zero, index out of bounds |
| MSan | 5 | MemorySanitizer, use-of-uninitialized-value |
| TSan | 7 | ThreadSanitizer, data race, lock-order-inversion, thread leak |
| Valgrind | 12 | Invalid read/write, uninitialised value, definitely lost |
| Crash | 16 | SIGSEGV, SIGABRT, core dumped, stack smashing, malloc corruption |

**Приоритет классификации:**

```
1. ASan/UBSan/MSan/TSan/Crash паттерны → CONFIRMED (conf ≥ 0.85)
2. Signal exit code (-11, 134, etc.)   → CONFIRMED (conf = 0.9)
3. Expected signals match              → CONFIRMED (conf = 0.85)
4. Valgrind invalid access             → CONFIRMED (conf = 0.85)
5. Valgrind memory leak / errors       → SUGGESTIVE (conf = 0.6)
6. Non-zero exit + suggestive patterns → SUGGESTIVE (conf = 0.5)
7. Non-zero exit (plain)               → SUGGESTIVE (conf = 0.3)
8. Timeout                             → NEUTRAL (conf = 0.0)
9. Clean exit, no patterns             → SAFE (conf = 1.0)
```

**Confidence scoring по sanitizer:**

| Sanitizer | Base Confidence |
|-----------|----------------|
| ASan | 0.95 |
| MSan | 0.93 |
| TSan | 0.91 |
| UBSan | 0.90 |
| Crash | 0.85 |
| Multi-source (>1 категория) | +0.05 boost |

#### 3.6.6. CWE Mapping

Система автоматически определяет идентификатор CWE из вывода санитайзера. Маппинг `SANITIZER_TO_CWE` содержит 23 записи:

| Sanitizer Output | CWE |
|------------------|-----|
| heap-buffer-overflow | CWE-122 |
| stack-buffer-overflow | CWE-121 |
| global-buffer-overflow | CWE-120 |
| heap-use-after-free | CWE-416 |
| double-free | CWE-415 |
| stack-use-after-return | CWE-562 |
| alloc-dealloc-mismatch | CWE-762 |
| container-overflow | CWE-787 |
| signed integer overflow | CWE-190 |
| division by zero | CWE-369 |
| null pointer | CWE-476 |
| use-of-uninitialized-value | CWE-457 |
| data race | CWE-362 |
| lock-order-inversion | CWE-833 |
| thread leak | CWE-404 |

### 3.7. Шаг 5: Агрегация вердикта

```
CONFIRMED  ← хотя бы один exploit подтверждён
SUGGESTIVE ← есть SUGGESTIVE evidence или hybrid_score > 0.7 без evidence
NEUTRAL    ← есть evidence, но всё SAFE, или hybrid_score ≤ 0.7
SAFE       ← hybrid_score < detection_threshold
```

**Confidence вычисляется по формуле:**
- CONFIRMED: `min(0.99, hybrid_score × 0.5 + max_evidence_conf × 0.5)`
- SUGGESTIVE: `hybrid_score × 0.9`
- NEUTRAL: `hybrid_score × 0.7`
- SAFE: `1 - hybrid_score`

**Early stop**: при включённом флаге `early_stop_on_confirmed` (по умолчанию True) пайплайн останавливает перебор exploit'ов после первого CONFIRMED, экономя время.

---

## 4. Детальное описание модулей

### 4.1. Модуль 1: SAST Detection (`code/`)

#### 4.1.1. Архитектура модели — `model.py`

**GNNReGVD** — гибридная модель, объединяющая предобученный GraphCodeBERT с графовыми нейросетями.

**LoRA (Low-Rank Adaptation):**
- Замораживает веса GraphCodeBERT (~125M параметров)
- Добавляет обучаемые адаптеры A (768×8) и B (8×768) к query/value проекциям
- Обучаемых параметров: ~1M вместо 125M
- `Output = original(x) + x @ A @ B × (alpha/rank)`

**Графовые нейросети — `modelGNN_updates.py`:**
- **ReGCN** (Residual Graph Convolution Network): `H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l)) + H^(l)`
- **ReGGNN** (Residual Gated Graph Neural Network): с gating механизмом
- **GGGNN** (Graph Gated Neural Network): GRU-based message passing
- Граф строится из AST + sliding window (window_size=5)

**Функции потерь — `losses.py`:**
- **SupervisedContrastiveLoss**: притягивает эмбеддинги одного класса, отталкивает разных; temperature=0.07
- **TripletMarginLoss**: anchor-positive-negative с hard mining; margin=0.3
- Итоговый loss: `L = L_cls + λ × L_contrastive`

#### 4.1.2. FAISS Index — `faiss_index.py`

**FAISSIndexManager** управляет индексом для k-NN поиска.

Два режима:
- **Flat** (IndexFlatIP): точный поиск для <100K эмбеддингов
- **IVF** (IndexIVFFlat): приближённый поиск для >100K

**Ключевой метод — `compute_vuln_type()`:**
Majority vote по типам уязвимостей среди k ближайших уязвимых соседей.

#### 4.1.3. Hybrid Inference — `inference.py`

**HybridPredictor** — двойной скоринг:

```
hybrid_score = β × cls_prob + (1 - β) × faiss_score
```

Параметр β подбирается на validation set через grid search (по умолчанию 0.6).

### 4.2. Модуль 2: Exploit Retrieval (`exploit_db/exploit_index.py`)

Отдельный FAISS-индекс для exploit-шаблонов. Два раздельных индекса (для детекции и эксплойтов) обеспечивают независимое обновление.

**Двухступенчатый поиск:**
1. FAISS similarity search (broad, top_k × 5)
2. Фильтрация по `vuln_type`, возврат top_k

### 4.3. Модуль 3: Exploit Adaptation (`exploit_db/exploit_adapter.py`)

#### 4.3.1. LLM-интеграция

| Backend | Модель | Использование |
|---------|--------|---------------|
| `transformers` | Qwen2.5-Coder-7B-Instruct (локально) | GPU с 16GB+ VRAM |
| `api` | Любая OpenAI-compatible модель | HTTP API (Ollama, vLLM, OpenAI) |

**Iterative Refinement (VBF-inspired):**
При неудаче LLM-генерации система сохраняет код и ошибку, формирует refinement prompt и повторяет запрос (до `max_retries` раз).

**Кэширование:**
Результаты LLM-адаптации кэшируются in-memory по ключу `MD5(vuln_code + vuln_type + exploit_id)`.

**Валидация синтаксиса:**
Перед отправкой в sandbox: `gcc -fsyntax-only harness.c` → `(is_valid, error_message)`.

### 4.4. Модуль 4: Sandbox Verification (`sandbox/`)

Подробно описан в разделе 3.6.

### 4.5. Taint Analysis (`analysis/taint.py`)

Подробно описан в разделе 3.2.

### 4.6. Pipeline Orchestrator (`scanner/pipeline.py`)

**VulnerabilityScanner** — главный класс, оркеструющий все модули. Lazy initialization: каждый модуль загружается при первом обращении.

```python
scanner = VulnerabilityScanner(ScannerConfig(
    model_path="saved_models/best.bin",
    faiss_dir="saved_models/faiss_index",
    exploit_db_dir="saved_models/exploit_db",
    llm_backend="api",
    llm_api_url="http://localhost:11434/v1/chat/completions",
    sandbox_timeout=30,
    sandbox_use_multi_sanitizer=True,
    sandbox_use_libfuzzer=True,
    use_taint_analysis=True,
))

result = scanner.scan(source_code)
# result.verdict:          "CONFIRMED" | "SUGGESTIVE" | "NEUTRAL" | "SAFE"
# result.confidence:       0.0 - 1.0
# result.vuln_type:        "buffer_overflow"
# result.cwe_id:           "CWE-122"
# result.sanitizer_source: "asan_ubsan"
# result.taint_context:    "Taint Analysis Results: ..."
```

---

## 5. Покрытие типов уязвимостей

### 5.1. Полная матрица покрытия

| Тип уязвимости | CWE | Шаблоны | Generic | Sanitizer | Taint Detect |
|----------------|-----|---------|---------|-----------|-------------|
| Buffer Overflow (stack) | CWE-121 | 3 | buffer_overflow | ASan | memory sinks |
| Buffer Overflow (heap) | CWE-122 | 2 | heap_buffer_overflow | ASan | memory sinks |
| Buffer Overflow (global) | CWE-120 | — | — | ASan | memory sinks |
| Use-After-Free | CWE-416 | 2 | use_after_free | ASan | alloc sinks |
| Double-Free | CWE-415 | — | double_free | ASan | alloc sinks |
| Format String | CWE-134 | 2 | format_string | ASan | format sinks |
| Integer Overflow | CWE-190 | 2 | integer_overflow | UBSan | — |
| Null Pointer Deref | CWE-476 | 2 | null_pointer_deref | ASan | — |
| Off-by-One | CWE-193 | 2 | off_by_one | ASan | memory sinks |
| Division by Zero | CWE-369 | 2 | division_by_zero | UBSan/SIGFPE | — |
| Uninitialized Read | CWE-457 | 2 | uninitialized_read | **MSan** | — |
| Data Race | CWE-362 | 1 | race_condition | **TSan** | — |
| Deadlock | CWE-833 | — | — | **TSan** | — |
| Thread Leak | CWE-404 | — | — | **TSan** | — |
| Stack Exhaustion | — | — | stack_exhaustion | SIGSEGV | — |
| Alloc/Dealloc Mismatch | CWE-762 | — | alloc_dealloc_mismatch | ASan | alloc sinks |
| Container Overflow | CWE-787 | — | — | ASan | — |
| Command Injection | CWE-78 | — | — | — | exec sinks |

**Итого:** 18 типов уязвимостей, 20 шаблонов, 15 generic-генераторов, 3 профиля санитайзеров, 23 CWE-маппинга.

---

## 6. Конфигурация

### 6.1. ScannerConfig — полный список параметров

| Группа | Параметр | Тип | Default | Описание |
|--------|----------|-----|---------|----------|
| Detection | `model_path` | str | "" | Путь к checkpoint модели |
| Detection | `faiss_dir` | str | "" | Путь к FAISS индексу |
| Detection | `tokenizer_name` | str | "microsoft/graphcodebert-base" | Токенизатор |
| Detection | `block_size` | int | 400 | Макс. длина входа в токенах |
| Detection | `hidden_size` | int | 128 | Размер скрытого слоя GNN |
| Detection | `num_GNN_layers` | int | 2 | Количество слоёв GNN |
| Detection | `gnn` | str | "ReGCN" | Тип GNN: ReGCN/ReGGNN/GGGNN |
| Detection | `use_lora` | bool | True | LoRA-адаптеры |
| Detection | `lora_rank` | int | 8 | Ранг LoRA |
| Detection | `embed_dim` | int | 512 | Размерность эмбеддинга |
| Scoring | `beta` | float | 0.6 | Вес classifier vs FAISS |
| Scoring | `top_k` | int | 5 | k для k-NN |
| Scoring | `detection_threshold` | float | 0.5 | Порог детекции |
| Exploit | `exploit_db_dir` | str | "" | Путь к exploit DB |
| Exploit | `exploit_top_k` | int | 3 | Количество exploit-шаблонов |
| LLM | `llm_backend` | str? | None | "transformers" / "api" / None |
| LLM | `llm_model_name` | str | "Qwen/Qwen2.5-Coder-7B-Instruct" | Модель LLM |
| LLM | `llm_api_url` | str? | None | URL API (Ollama, vLLM, etc.) |
| LLM | `llm_api_key` | str? | None | API ключ (если требуется) |
| LLM | `llm_max_retries` | int | 3 | Макс. попыток LLM |
| LLM | `llm_temperature` | float | 0.0 | Temperature (0 = deterministic) |
| LLM | `llm_cache_results` | bool | True | Кэширование LLM результатов |
| Sandbox | `sandbox_memory` | str | "1g" | Лимит RAM |
| Sandbox | `sandbox_cpu` | float | 0.9 | Лимит CPU |
| Sandbox | `sandbox_timeout` | int | 30 | Timeout в секундах |
| Sandbox | `sandbox_network` | str | "none" | Сеть контейнера |
| Sandbox | `sandbox_docker_image` | str | "" | Docker образ |
| Sandbox | `sandbox_use_valgrind` | bool | False | Valgrind |
| Sandbox | `sandbox_use_seccomp` | bool | True | Seccomp-профиль |
| **Sandbox** | **`sandbox_use_multi_sanitizer`** | **bool** | **False** | **Multi-sanitizer (ASan+MSan+TSan)** |
| **Sandbox** | **`sandbox_use_libfuzzer`** | **bool** | **False** | **LibFuzzer** |
| **Sandbox** | **`sandbox_fuzz_time`** | **int** | **15** | **Время фаззинга (сек)** |
| **Sandbox** | **`sandbox_prefer_clang`** | **bool** | **False** | **Предпочитать clang** |
| **Taint** | **`use_taint_analysis`** | **bool** | **False** | **Taint analysis** |
| **Taint** | **`joern_path`** | **str?** | **None** | **Путь к Joern CLI** |
| Pipeline | `max_exploits_per_sample` | int | 3 | Макс. harness на сэмпл |
| Pipeline | `early_stop_on_confirmed` | bool | True | Остановка при CONFIRMED |
| Pipeline | `language` | str | "c" | Язык исходного кода |
| Pipeline | `use_multi_harness` | bool | True | Multi-harness генерация |

---

## 7. Технологический стек

### 7.1. ML/DL фреймворки

| Библиотека | Назначение |
|------------|------------|
| PyTorch | Обучение и инференс GNN модели |
| Transformers (HuggingFace) | GraphCodeBERT, LoRA, токенизация |
| FAISS (faiss-cpu/faiss-gpu) | Approximate Nearest Neighbor поиск |
| NumPy | Операции с эмбеддингами |

### 7.2. LLM

| Модель | Назначение |
|--------|------------|
| Qwen2.5-Coder-7B-Instruct | Локальный LLM для адаптации эксплойтов |
| Любая OpenAI-compatible API | Альтернативный LLM бэкенд (Ollama, vLLM) |

### 7.3. Инфраструктура

| Технология | Назначение |
|------------|------------|
| Docker | Изоляция sandbox |
| Seccomp | Фильтрация системных вызовов |
| GCC 14 | Компиляция с ASan + UBSan |
| Clang 18 | MSan, TSan, LibFuzzer (coverage-guided fuzzing) |
| AddressSanitizer (ASan) | Детекция memory errors |
| UndefinedBehaviorSanitizer (UBSan) | Детекция undefined behavior |
| MemorySanitizer (MSan) | Детекция неинициализированных чтений |
| ThreadSanitizer (TSan) | Детекция гонок данных и deadlock'ов |
| LibFuzzer | Coverage-guided fuzzing |
| Valgrind | Вторичная проверка памяти |
| Joern | Статический taint analysis через CPG |

---

## 8. Развёртывание и запуск

### 8.1. Установка зависимостей

```bash
# Обязательные (core)
pip install torch>=2.0.0 transformers>=4.30.0 numpy>=1.24.0 pytest>=7.0.0

# FAISS (для Модулей 1, 2)
pip install faiss-cpu>=1.7.4    # или faiss-gpu для GPU

# LLM (только при llm_backend="transformers")
pip install accelerate>=0.20.0 sentencepiece>=0.1.99
```

**Системные зависимости:**

```bash
# macOS:
brew install gcc                   # ASan + UBSan
brew install llvm                  # MSan, TSan, LibFuzzer (clang)
brew install --cask docker         # sandbox (опционально)

# Ubuntu/Debian:
sudo apt-get install gcc clang valgrind docker.io

# Joern (опционально, для taint analysis):
# https://joern.io/install
curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" | bash
```

### 8.2. Установка

```bash
# 1. Клонирование
git clone <repository-url>
cd GNN-ReGVD

# 2. Создание virtualenv
python -m venv .venv
source .venv/bin/activate

# 3. Установка зависимостей
pip install torch transformers numpy faiss-cpu pytest

# 4. Проверка: запуск тестов (не требует GPU/Docker/Joern)
python -m pytest tests/ -v --ignore=tests/test_faiss_vuln_type.py

# 5. (Опционально) Сборка Docker образов
docker build -t gnn-regvd-sandbox:c-cpp \
    -f sandbox/dockerfiles/Dockerfile.c_cpp sandbox/dockerfiles/
docker build -t gnn-regvd-sandbox:clang \
    -f sandbox/dockerfiles/Dockerfile.clang sandbox/dockerfiles/
```

### 8.3. Режимы запуска

#### Sandbox-only (без обученной модели, без GNN)

Прямая верификация кода через exploit-адаптацию и sandbox:

```bash
# Код из командной строки
python scan_cli.py --sandbox-only \
    --code "void f(const char *s){char b[8];strcpy(b,s);}"

# Файл
python scan_cli.py --sandbox-only --file vuln.c --verbose

# С LLM адаптацией через Ollama
python scan_cli.py --sandbox-only \
    --code "void f(char *s){char b[8];strcpy(b,s);}" \
    --llm-backend api \
    --llm-api-url "http://localhost:11434/v1/chat/completions" \
    --llm-model "qwen2.5-coder:7b"
```

#### Полный pipeline (требует обученную GNN-модель)

```bash
# 1. Обучение модели
cd code
python run.py --do_train --do_eval --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --output_dir=./saved_models/lora_faiss \
    --use_lora --use_faiss --embed_dim 512 \
    --num_train_epochs 10

# 2. Сканирование
cd ..
python scan_cli.py --file vuln.c \
    --model-path code/saved_models/lora_faiss/checkpoint-best-acc/model.bin \
    --faiss-dir code/saved_models/lora_faiss/faiss_index \
    --verbose
```

#### С multi-sanitizer + LibFuzzer + taint analysis

```bash
python scan_cli.py --file vuln.c \
    --model-path code/saved_models/lora_faiss/checkpoint-best-acc/model.bin \
    --faiss-dir code/saved_models/lora_faiss/faiss_index \
    --multi-sanitizer \
    --use-libfuzzer --fuzz-time 15 \
    --taint-analysis \
    --prefer-clang \
    --verbose
```

#### Batch-сканирование датасета

```bash
# Первые 50 сэмплов из test.jsonl
python scan_cli.py --jsonl dataset/test.jsonl \
    --limit 50 --output results.json --verbose

# Оценка метрик
python evaluate.py --mode sandbox-only --limit 20 --timeout 10
```

### 8.4. CLI-флаги

| Флаг | Описание |
|------|----------|
| `--code TEXT` | C/C++ код для сканирования |
| `--file PATH` | Файл с исходным кодом |
| `--jsonl PATH` | JSONL-датасет для batch-сканирования |
| `--sandbox-only` | Режим без GNN (только sandbox verification) |
| `--model-path PATH` | Путь к checkpoint GNN-модели |
| `--faiss-dir PATH` | Путь к FAISS-индексу |
| `--llm-backend {transformers,api}` | Бэкенд LLM |
| `--llm-model NAME` | Имя модели LLM |
| `--llm-api-url URL` | URL API для LLM |
| `--threshold FLOAT` | Порог детекции (default: 0.5) |
| `--max-exploits INT` | Макс. exploit-попыток (default: 3) |
| `--sandbox-timeout INT` | Timeout sandbox в секундах (default: 30) |
| `--use-valgrind` | Включить Valgrind |
| **`--multi-sanitizer`** | **Multi-sanitizer (ASan+UBSan, MSan, TSan)** |
| **`--use-libfuzzer`** | **LibFuzzer (coverage-guided fuzzing)** |
| **`--fuzz-time INT`** | **Время фаззинга в секундах (default: 15)** |
| **`--taint-analysis`** | **Taint analysis (Joern/heuristic)** |
| **`--prefer-clang`** | **Предпочитать clang для компиляции** |
| `--limit INT` | Лимит сэмплов для batch-режима |
| `--output PATH` | Файл для сохранения результатов (JSON) |
| `--verbose, -v` | Подробный вывод |
| `--json` | Вывод в формате JSON |

---

## 9. Тестирование

### 9.1. Общая статистика

**203 теста**, время выполнения ~6 секунд.

Тесты не требуют GPU, Docker, Joern или обученной модели (всё мокируется).

```bash
python -m pytest tests/ -v --ignore=tests/test_faiss_vuln_type.py
```

### 9.2. Разбивка по файлам

| Файл | Тестов | Покрытие |
|------|--------|----------|
| test_sandbox.py | 61 | Evidence classification, confidence, ASan, Valgrind, docker, batch execution |
| test_enhancements.py | 55 | MSan, TSan, CWE mapping, multi-sanitizer profiles, LibFuzzer harness, taint analysis, new generic harnesses, new templates |
| test_adapter.py | 36 | LLM retry, JSON parsing, multi-harness, cache, syntax validation |
| test_pipeline.py | 25 | Pipeline orchestration, verdict aggregation, confidence, CLI, e2e |
| test_evaluate.py | 18 | Metrics computation, dataset loading, confusion matrix |
| test_exploit_db.py | 7 | ExploitDBManager (требует faiss) |
| test_faiss_vuln_type.py | 4 | compute_vuln_type (требует faiss) |

### 9.3. test_enhancements.py — 55 тестов (новые)

| Группа | Тестов | Описание |
|--------|--------|----------|
| TestMsanEvidence | 3 | MSan → CONFIRMED, heap alloc detection, CWE-457 mapping |
| TestTsanEvidence | 4 | Data race, lock-order-inversion, CWE-362, thread leak |
| TestCweMapping | 7 | CWE для heap overflow, UAF, div zero, int overflow; sanitizer source; dict coverage |
| TestMultiSanitizer | 10 | Profile definitions, gcc/clang requirements, fuzz profiles, vuln_type selection |
| TestFuzzHarnessWrapping | 4 | main()→LLVMFuzzerTestOneInput, no-main rejection, header add/dedup |
| TestNewGenericHarnesses | 7 | off_by_one, division_by_zero, uninitialized_read, race_condition, stack_exhaustion, alloc_dealloc_mismatch; hint coverage |
| TestTaintAnalyzer | 9 | Heuristic detection: strcpy/system sinks, taint paths, safe code, format_for_llm, caller chains, joern fallback |
| TestScannerConfigNewFields | 2 | Default values, override values |
| TestScanResultNewFields | 2 | cwe_id, taint_context fields |
| TestNewTemplateFiles | 7 | Existence of all 10 template files, total count ≥ 20 |

---

## 10. Формат данных

### 10.1. Входные данные (`test.jsonl`)

```json
{"project": "FFmpeg", "commit_id": "32bf6550...", "target": 0, "func": "int ff_get_wav_header(...) { ... }", "idx": 3}
```

Датасет: 2 732 сэмпла (1 477 safe / 1 255 vulnerable).

### 10.2. Выходные данные (ScanResult)

```json
{
    "verdict": "CONFIRMED",
    "confidence": 0.95,
    "cls_prob": 0.9,
    "faiss_score": 0.85,
    "hybrid_score": 0.88,
    "vuln_type": "buffer_overflow",
    "vuln_type_confidence": 0.8,
    "cwe_id": "CWE-122",
    "sanitizer_source": "asan_ubsan",
    "taint_context": "Taint Analysis Results:\n  1. gets(line 5) -[direct]-> strcpy(line 8) [memory]",
    "exploits_tried": 2,
    "exploits_confirmed": 1,
    "evidence_details": [
        {
            "exploit_id": "BO-STRCPY-001",
            "evidence_level": "CONFIRMED",
            "reason": "Direct exploitation evidence: ASAN: AddressSanitizer",
            "exit_code": 1,
            "execution_time": 0.52,
            "compiled": true,
            "confidence": 0.95,
            "cwe_id": "CWE-122",
            "sanitizer_source": "asan_ubsan",
            "sanitizer_profile": "asan_ubsan",
            "fuzz_mode": false
        }
    ],
    "total_time": 1.5
}
```

---

## 11. Метрики оценки

| Метрика | Формула | Целевое значение |
|---------|---------|-----------------|
| Accuracy | (TP+TN) / Total | >85% |
| Precision | TP / (TP+FP) | >80% |
| Recall | TP / (TP+FN) | >80% |
| F1 | 2·P·R / (P+R) | >80% |
| FP Reduction | (FP_before - FP_after) / FP_before | >50% |
| Exploitation Rate | Confirmed / Total_tried | >60% |
| Pipeline Latency | Среднее время на сэмпл | <60s |

---

## 12. Безопасность

### 12.1. Изоляция sandbox

- **Сетевая изоляция**: `--network=none` полностью блокирует сеть
- **Filesystem**: read-only root, tmpfs для /tmp и /out
- **Привилегии**: `--cap-drop=ALL`, `--security-opt=no-new-privileges`
- **Ресурсы**: memory limit 1g, CPU quota 0.9, PID limit 128
- **Seccomp**: whitelist из 70+ разрешённых syscalls
- **Пользователь**: непривилегированный `sandbox` user

### 12.2. Защита от LLM Prompt Injection

- System prompt hardened (фиксированная роль security researcher)
- Temperature = 0 (детерминированный вывод)
- Валидация output: JSON parsing + syntax check
- Максимум 3 попытки refinement
- Результат LLM никогда не выполняется напрямую — только через sandbox

---

## 13. Ограничения и будущие направления

### 13.1. Текущие ограничения

- Только C/C++ (расширяемо через Tree-sitter + новые Dockerfile)
- MSan и TSan требуют clang (не поддерживаются gcc)
- LibFuzzer работает только с clang
- Docker overhead ~2–5с на запуск контейнера
- Taint analysis через Joern требует установки отдельного инструмента

### 13.2. Возможные расширения

- Python/Java sandbox (Dockerfile.python, Dockerfile.java)
- Расширение Exploit DB из CVEfixes, PrimeVul, DiverseVul
- Container pool для снижения Docker overhead
- GPU batching для LLM inference
- AFL++ интеграция для альтернативного фаззинга
- Symbolic execution (KLEE) для генерации exploit-inputs
