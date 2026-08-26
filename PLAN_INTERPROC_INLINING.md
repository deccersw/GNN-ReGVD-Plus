# Module 0: Interprocedural Inlining — план реализации

Статус: **черновик плана, реализация не начата**
Дата: 2026-08-15
Источник идеи: `Межфайловое взаимодействие v1.md` (depth-limited inlining перед Module 1)

---

## 0. TL;DR

Добавляем перед Module 1 новый модуль `interproc/`, который:

1. принимает на вход **директорию проекта** (C/C++), а не одну функцию;
2. строит индекс символов и call graph всего проекта;
3. для каждой функции-корня `F` строит **AnalysisUnit** — расширенное представление
   с телами вызываемых функций, подставленными до глубины `D` и в рамках токенного бюджета;
4. отдаёт список юнитов дальше в существующий пайплайн без изменения архитектуры
   GNN / FAISS / LoRA / LLM / sandbox.

Ключевое отклонение от исходной идеи (обосновано ниже, §2):
**каждый юнит несёт два представления кода**, а не одно —
`code_for_gnn` (инлайненный текст под 398-токенный бюджет GraphCodeBERT) и
`code_for_sandbox` (компилируемый bundle из полных определений зависимостей).
Одно представление не может обслужить одновременно GNN и sandbox.

---

## 1. Что мы знаем про текущий пайплайн (факты из кода)

Разбор проведён по коду, не по документации. Существенные для дизайна факты:

| Факт | Где | Следствие для Module 0 |
|---|---|---|
| Вход `scan(source_code: str)` — одна строка кода | `scanner/pipeline.py:100` | Интерфейс юнита должен сводиться к строке (+ метаданные) |
| `predict_single` делает `' '.join(code.split())`, затем `tokenizer.tokenize(code)[:block_size-2]` | `code/inference.py:80-85` | **Жёсткая обрезка на 398 BPE-токенов.** Всё, что не влезло — молча выбрасывается |
| `block_size = 400` | `scanner/config.py:16` | ~398 BPE-токенов ≈ 60–100 строк C. Инлайнинг мгновенно упирается в потолок |
| Комментарии **не** вырезаются перед токенизацией | `code/run.py:94-101` | Комментарии-маркеры инлайна съедают бюджет и создают OOD-токены → в `code_for_gnn` их по умолчанию нет |
| Граф строится по `build_graph(format="uni")`: узел = **уникальный токен** документа, рёбра = со-встречаемость в окне 5 | `code/model.py:262`, `code/modelGNN_updates.py:204` | Повторный инлайн одной функции не удваивает узлы, а усиливает рёбра. **Alpha-renaming увеличивает число уникальных токенов** → раздувает граф и уводит распределение от обучающего |
| Модель обучена на одиночных функциях Devign | `dataset/*.jsonl`, `code/run.py` | Инлайненные функции — сдвиг распределения. Нужна калибровка/дообучение (§9) |
| FAISS-индекс построен по эмбеддингам не-инлайненных функций | `code/faiss_index.py` | Соседи для инлайненного юнита могут «поехать» → опция пересборки индекса |
| `TaintAnalyzer` уже умеет call graph и `FunctionInfo(name, start, end)` через Joern или regex | `analysis/taint.py:324-398` | Есть готовый fallback-парсер и список `DANGEROUS_SINKS` / `INPUT_SOURCES` — переиспользуем |
| Пайплайн строится по каскадам с деградацией (LLM→template→generic; joern→heuristic) | `exploit_db/exploit_adapter.py`, `analysis/taint.py` | Module 0 делаем в том же идиоме: tree-sitter → regex fallback |
| `ScannerConfig` — плоский dataclass, без YAML | `scanner/config.py` | Новые параметры добавляем с префиксом `inline_*`, YAML — опциональный лоадер поверх |
| `scan_jsonl` фильтрует поля по `ScanResult.__dataclass_fields__` | `scan_cli.py:172` | Новые поля `ScanResult` с дефолтами не ломают существующий код и тесты |
| В окружении есть: `pycparser`, `networkx`, `yaml`, `gcc`, `clang`, `ctags`. Нет: `tree_sitter`, `libclang` | проверено | Выбор парсера — см. §3 |

---

## 2. Ключевое проектное решение: два представления кода

Исходная идея говорит «подставили тело в AST и погнали дальше по пайплайну».
Проблема: дальше по пайплайну **два разных потребителя с несовместимыми требованиями**.

**Потребитель A — Module 1 (GNN).** Ему нужна только последовательность токенов.
Семантическая корректность C не требуется вообще: код не компилируется,
граф строится по со-встречаемости токенов. Жёсткое ограничение — 398 BPE-токенов.

**Потребитель B — Module 3/4 (LLM-адаптация → sandbox).** Ему нужен код, который
**реально компилируется**: LLM генерирует harness вокруг `vuln_code`, harness идёт
в `gcc/clang -fsanitize=...`. Если подать семантически битый инлайненный блоб
(оборванные `return` в середине, несвязанные типы, отсутствующие `#include`),
harness не соберётся, `harness_compiled=False`, и все вердикты выродятся в NEUTRAL —
то есть мы своими руками сломаем главный вклад работы (execution-based verification).

Отсюда: `AnalysisUnit` содержит **оба** представления.

```
                 ┌─ code_for_gnn      → Module 1 (GNN + FAISS)
AnalysisUnit ────┤
                 └─ code_for_sandbox  → Module 3 (LLM) → Module 4 (sandbox), Triage
                 + source_map / provenance → отчёт, дедупликация findings
```

- `code_for_gnn` — инлайненный текст. Семантика приблизительная, приоритет —
  плотность полезных токенов в бюджете 398.
- `code_for_sandbox` — **не инлайн**, а *bundle*: `#include` системных заголовков,
  нужные typedef/struct/enum/#define, глобалы, затем **полные определения callee**
  (в порядке «листья → корень»), затем сама `F`. Это одновременно
  (а) компилируемо, (б) семантически точно, (в) не требует переписывания `return`.
  Бюджет отдельный (контекст LLM, ~8–16k токенов), гораздо мягче.

Побочный бонус: почти все «страшные» краевые случаи инлайнинга
(early return, goto/labels, varargs, longjmp) для потребителя B просто не возникают,
а для потребителя A допустимы, потому что там семантика не проверяется.
Это резко снижает риск всей затеи.

---

## 3. Выбор парсера

| Вариант | Плюсы | Минусы | Решение |
|---|---|---|---|
| **tree-sitter** (`tree-sitter`, `tree-sitter-c`, `tree-sitter-cpp`) | error-recovery на непрепроцессированном коде, байтовые оффсеты, быстрый (~МБ/с), не нужны заголовки и флаги компиляции | новая зависимость | **Primary backend** |
| libclang | точный AST, разрешение типов и перегрузок | требует `compile_commands.json`/полные include-пути; на произвольном репозитории падает | Отклонён (может быть backend v2) |
| pycparser | есть в окружении | только препроцессированный чистый C89/99, без C++ | Отклонён |
| Joern (уже опционально используется) | настоящий CPG, точный call graph, межфайловый | JVM, минуты на проект, опциональная зависимость | **Optional backend** для уточнения рёбер (этап 8) |
| regex (как в `analysis/taint.py`) | ноль зависимостей, уже написан | пропускает функции, не понимает вложенность | **Fallback backend**, нужен чтобы модуль работал «из коробки» |

Каскад: `tree_sitter → regex`, выбор через `inline_parser_backend: auto|treesitter|regex|joern`.
`auto` = tree-sitter если импортируется, иначе regex + WARNING в лог.
Все тесты пишем так, чтобы regex-backend проходил тот же контракт (и помечаем
tree-sitter-тесты `pytest.mark.skipif`), — иначе CI сломается у соавторов.

---

## 4. Структура модуля

```
interproc/
  __init__.py
  config.py              InliningConfig (dataclass)
  models.py              FunctionDef, CallSite, CallEdge, TypeDef, GlobalDef,
                         MacroDef, Segment, SourceMap, AnalysisUnit, ProjectIndex
  discovery.py           обход проекта, фильтры, детект языка, кэш по mtime+hash
  parsers/
    base.py              ParserBackend (Protocol): parse_file() -> FileFacts
    treesitter_backend.py
    regex_backend.py     переиспользует эвристики analysis/taint.py
    joern_backend.py     (этап 8, опционально)
  symbols.py             таблица символов проекта, include-граф
  callgraph.py           разрешение вызовов, рёбра, SCC, экспорт в DOT
  budget.py              токенный бюджет (реальный GraphCodeBERT BPE, кэш; fallback-оценка)
  scoring.py             приоритет call-site (sink / taint / размер / глубина)
  inliner.py             depth-limited expansion, renaming, rewriting, source map
  bundler.py             компилируемый bundle зависимостей
  units.py               сборка AnalysisUnit, дедуп, JSONL сериализация
  stats.py               метрики покрытия/разрешения/роста/обрезки
  cli.py                 python -m interproc.cli --project ...
scanner/
  project_scanner.py     (этап 7) scan_project + агрегация/дедуп findings
tests/
  test_interproc_discovery.py
  test_interproc_callgraph.py
  test_interproc_inliner.py
  test_interproc_bundler.py
  test_interproc_cli.py
test_samples/projects/   синтетические мини-проекты p01..p08 (§10)
```

Именование модуля: `interproc` (не `inlining`), т.к. со временем туда же логично
переедет межпроцедурный taint и slicing.

---

## 5. Модели данных

```python
@dataclass(frozen=True)
class FuncId:                     # стабильный ключ функции в проекте
    file: str                     # относительный путь от корня проекта
    name: str
    start_line: int
    # __str__ -> "src/net/parse.c:handle_packet:120"

@dataclass
class FunctionDef:
    fid: FuncId
    name: str
    qualified_name: str           # C++: Ns::Class::method
    return_type: str
    params: list[Param]           # (type, name); is_vararg отдельно
    is_static: bool               # ограничивает видимость файлом (TU)
    is_vararg: bool
    is_virtual: bool              # C++
    is_template: bool             # C++
    signature_text: str
    body_text: str                # включая внешние { }
    start_line: int; end_line: int
    start_byte: int; end_byte: int
    token_count: int              # BPE, лениво
    body_hash: str                # для дедупа одинаковых тел
    calls: list[CallSite]
    flags: set[str]               # has_goto, has_label, has_setjmp, has_asm,
                                  # has_preproc_branch, unbalanced_preproc

@dataclass
class CallSite:
    callee_name: str
    arg_texts: list[str]
    arg_count: int
    start_byte: int; end_byte: int
    line: int
    kind: str                     # direct | method | fptr | macro | unknown
    receiver_text: str = ""       # для obj->m()

@dataclass
class CallEdge:
    caller: FuncId
    callee: FuncId | None         # None => не разрешено
    site: CallSite
    resolution: str               # exact_same_file | exact_include | exact_global |
                                  # arity_filtered | fptr_single_assign |
                                  # ambiguous | external | macro | unresolved
    confidence: float

@dataclass
class Segment:                    # кусок итогового текста с происхождением
    text: str
    origin_fid: FuncId | None
    origin_line_start: int
    depth: int
    role: str                     # root | param_bind | body | ret_assign | marker

@dataclass
class AnalysisUnit:
    unit_id: str                  # sha1(root_fid + config_hash)
    root: FuncId
    code_for_gnn: str
    code_for_sandbox: str
    segments: list[Segment]       # -> source_map
    inlined: list[InlineRecord]   # (fid, depth, tokens_added, reason_score)
    skipped: list[SkipRecord]     # (callee_name, reason)
    depth_used: int
    tokens_gnn: int
    truncated: bool               # tokens_gnn упёрлось в лимит
    provenance_files: list[str]
    stats: dict
```

Сериализация юнита в JSONL — **совместимая с существующим форматом датасета**
(`{"func": ..., "target": ..., "idx": ...}`), чтобы юниты можно было прогонять
через `scan_cli.py --jsonl` и `evaluate.py` без правок:

```json
{"idx": 0, "func": "<code_for_gnn>", "target": -1,
 "unit": {"unit_id": "...", "root": "src/a.c:f:10", "code_for_sandbox": "...",
          "inlined": [...], "skipped": [...], "depth_used": 2,
          "tokens_gnn": 391, "truncated": false, "provenance_files": [...]}}
```

---

## 6. Алгоритм

### 6.1. Discovery
- Расширения: `.c .h .cc .cpp .cxx .c++ .hpp .hh .hxx .inl`.
- Исключения по умолчанию: `build/ cmake-build*/ .git/ node_modules/ third_party/
  vendor/ external/ .venv/ dist/ out/`, плюс `inline_exclude_globs` из конфига.
- Лимиты: `max_file_bytes` (по умолчанию 2 МБ), `max_files`.
- Symlink-петли: `os.walk(followlinks=False)` + set посещённых `realpath`.
- Кодировки: читаем `utf-8` с `errors="replace"`; бинарные файлы (NUL в первых 8 КБ) — skip.
- CRLF нормализуем при чтении (иначе поедут байтовые оффсеты и номера строк).
- Кэш парсинга: ключ `(path, size, mtime_ns, sha1[:16], parser_version)` → `FileFacts`,
  на диске в `.interproc_cache/`. Обязателен для итеративной работы на больших репо.

### 6.2. Symbol table + include-граф
- Таблица `name -> [FunctionDef]` по всему проекту.
- Include-граф: `#include "x.h"` резолвим (а) относительно каталога файла,
  (б) по `inline_include_dirs`, (в) по совпадению basename в проекте.
  `#include <x>` — системный, игнорируем при резолве, но сохраняем для bundler.
- Транзитивное замыкание include для каждого `.c` — это «область видимости» файла.

### 6.3. Разрешение вызовов (по приоритету правил)
1. Определение в **том же файле** → берём его (покрывает `static`-хелперы).
2. Кандидат `is_static=True` из **другого** файла — **запрещён** (другой TU),
   кроме случая «единственный кандидат во всём проекте» под флагом
   `inline_allow_cross_tu_static` (по умолчанию `false`).
3. Кандидаты, чей файл входит в include-замыкание вызывающего файла — приоритетнее прочих.
4. Фильтр по арности (с учётом varargs и default-аргументов C++).
5. Остался ровно один → `exact_*`. Осталось >1 → `ambiguous`, **не инлайним**.
   Остался 0 → `external` (объявлена, но не определена в проекте) или `unresolved`.
6. `obj.m()/obj->m()`: ищем метод по имени среди классов; >1 класс → `ambiguous`.
   `is_virtual` → **никогда не инлайним** (динамическая диспетчеризация неразрешима статически).
7. Указатели на функции: `unresolved` по умолчанию; опция
   `inline_resolve_single_assign_fptr` — если в теле вызывающей функции ровно одно
   присваивание `fp = &foo;` / `fp = foo;`, резолвим в `foo` с `confidence=0.6`.
8. Имя есть в таблице макросов → `macro`, не инлайним (но помечаем в stats).
9. **Never-inline список** (жёстко): все имена из `analysis/taint.py:DANGEROUS_SINKS`
   и `INPUT_SOURCES`, libc/STL (`str*`, `mem*`, `malloc/free/realloc`, `printf`-семейство,
   `open/read/write`, `std::*`) + пользовательский `inline_never_list`.
   **Обоснование, важное для статьи:** эти токены — ровно те признаки, на которых
   обучена модель. Заинлайнить `strcpy` (даже если его определение есть в проекте) =
   уничтожить сигнал. Инлайним только «свои» функции проекта.

Инварианты: разрешение **детерминировано** — при равных весах сортировка по
`(file, start_line)`. Воспроизводимость нужна для ablation в статье.

### 6.4. Depth-limited expansion (ядро)

Для корня `F`:

```
expand(F, depth=0, path=[F], budget)
  candidates = [site for site in F.calls if edge(site).callee is not None]
  score candidates (§6.5), sort desc, deterministic tie-break
  for site in candidates:
      g = edge(site).callee
      if depth >= max_depth:                    skip("depth")
      if g in path:                             skip("recursion")      # прямая и взаимная
      if occurrences[g] >= max_expansions:      skip("repeat")
      if g.token_count > max_callee_tokens:     skip("callee_too_big")
      if g.name in never_inline:                skip("never_inline")
      if g.flags & {has_asm, has_setjmp} :      skip("unsafe_construct")
      if g.is_vararg:                           skip("vararg")
      body = memo[(g.fid, max_depth-depth)] or expand(g, depth+1, path+[g], ...)
      cost = tokens(param_binds) + tokens(body)
      if budget.remaining < cost:               skip("budget"); continue  # НЕ вставляем частично
      splice(site -> param_binds + body + ret_assign)
      budget.spend(cost); occurrences[g] += 1
```

Существенные детали:

- **Никаких частичных вставок.** Если тело не влезает целиком — оставляем вызов как есть.
  Иначе получим оборванный мусорный хвост прямо перед обрезкой токенизатором.
- **Бюджет считается настоящим токенизатором** GraphCodeBERT (`RobertaTokenizer`),
  а не эвристикой по символам: расхождение BPE на C-коде достигает 1.5–2×.
  Токенизатор загружается один раз, результаты кэшируются по `body_hash`.
  Fallback (если transformers недоступен в standalone-режиме): `len(text.split()) * 1.6`,
  с явным warning'ом.
- **Мемоизация** `(fid, remaining_depth) -> expanded_body`. Без неё project-level скан
  экспоненциален на diamond-графах; с ней — линеен по числу функций.
- **SCC** call-графа считаем заранее (`networkx`, уже есть в окружении); рёбра внутри
  SCC инлайним не более одного раза вдоль пути.
- **Подстановка вместо вызова** делается **через сегменты, а не `str.replace`**:
  текст тела разрезается по байтовым оффсетам call-site'ов и собирается как
  `list[Segment]`. Это (а) корректно при повторяющихся текстах вызовов,
  (б) бесплатно даёт source map.
- **Связывание параметров:** `T p = <arg_text>;` по типам из сигнатуры callee.
  Если тип не распознан — `__typeof__` не используем (не портабельно), пишем сырое
  присваивание `p = <arg>;` (для GNN-представления этого достаточно).
- **`return`:** если значение вызова используется (call-site не в statement-позиции) —
  `return expr;` → `__r_<g> = expr;` и call-site заменяется на `__r_<g>`,
  с объявлением `T __r_<g>;` перед блоком. Если не используется — `return expr;` → `expr;`.
  Ранние `return` **ломают поток управления** — для GNN это допустимо (см. §2),
  для sandbox мы вообще не инлайним. Опция `inline_return_guard=goto` добавляет
  `goto __end_<g>;` + метку — включена по умолчанию `false` (лишние токены).
- **Переименование локальных переменных — только при реальной коллизии** с именами,
  живыми в области вставки. Причина в §1: в режиме `format="uni"` узел графа =
  уникальный токен, а `buf__inl1` даёт лишние BPE-подтокены и уводит распределение.
  Суффикс короткий и детерминированный: `_i1`, `_i2`.
- **Метки/goto в callee:** если `has_label` — либо skip, либо переименование меток
  (`L: -> L_i1:`); по умолчанию skip (`inline_skip_labeled_callees=true`).
- **Маркеры-комментарии** (`/* inlined g() */`) в `code_for_gnn` по умолчанию
  **выключены** — они съедают дефицитный бюджет и являются OOD-токенами.
  Вся эта информация есть в `segments`/`inlined`. Флаг `inline_debug_markers=true`
  включает их для ручной отладки.

### 6.5. Приоритезация call-site (что инлайнить первым)

Наивный DFS «сверху вниз» тратит бюджет на первый попавшийся крупный callee.
При потолке 398 токенов это критично. Поэтому вводим стратегии
(`inline_strategy = dfs | bfs | priority`, по умолчанию `priority`):

```
score(site, g, depth) =
      w_sink   * has_dangerous_sink(g)              # DANGEROUS_SINKS в теле g
    + w_mem    * has_memory_op(g)                   # malloc/free/индексная запись/указательная арифметика
    + w_taint  * arg_derived_from_params_or_input(site)   # аргумент выведен из параметров F или INPUT_SOURCES
    + w_small  * (1 - min(1, g.tokens / max_callee_tokens))
    + w_loop   * inside_loop(site)
    - w_depth  * depth
    - w_trivial* is_trivial_getter(g)               # тело = один return поля/константы
```

Веса — в конфиге, дефолты подбираем на мини-проектах, затем на Juliet.
Это, по сути, «depth cap + relevance-guided budget allocation», и это
самостоятельный вклад относительно голого depth-limited inlining
(есть что написать в статье и что заablate'ить).

### 6.6. Bundler (компилируемое представление)

Сбор `code_for_sandbox` для корня `F` и множества транзитивно достижимых callee `G`
(по тем же ограничениям глубины, но с отдельным, гораздо большим бюджетом):

1. Системные `#include <...>` из всех задействованных файлов — объединить, дедуп,
   отсортировать.
2. `#define`-константы и функциональные макросы, чьи имена встречаются в телах.
3. Типы: struct/union/enum/typedef, транзитивно достижимые из сигнатур и тел
   (замыкание по идентификаторам против таблицы типов проекта). Порядок —
   топологическая сортировка по зависимостям; при цикле — forward-декларации.
4. Глобальные переменные, встречающиеся в телах (с инициализаторами; `extern` — как есть).
5. Определения callee в порядке «листья → корень», затем `F`.
6. Для нерезолвнутых внешних символов — ничего не выдумываем; если удастся вывести
   сигнатуру из объявления в заголовке, добавляем прототип.
7. Ограничение `inline_bundle_max_tokens` (по умолчанию 12000); при переполнении —
   отбрасываем callee по возрастанию score из §6.5.

Метрика качества bundler'а: доля юнитов, проходящих `gcc -fsyntax-only`
(и `clang -fsyntax-only` для C++). Целевой ориентир — ≥80% на мини-проектах,
на реальных репо будет ниже, это честно фиксируем в статистике.

---

## 7. Интеграция с пайплайном (этап 7, после standalone-проверки)

Принцип: **не менять сигнатуры существующих методов**, только добавлять.

```python
# scanner/pipeline.py
def scan(self, source_code: str) -> ScanResult:          # без изменений
def scan_unit(self, unit) -> ScanResult:                 # НОВОЕ
    # _detect(unit.code_for_gnn)
    # _adapt_exploits(unit.code_for_sandbox, ..., taint_context + inline_context)
    # _triage(unit.code_for_sandbox, ...)
    # заполняет новые поля ScanResult
```

```python
# scanner/project_scanner.py (новый файл)
class ProjectScanner:
    def scan_project(self, path) -> ProjectScanResult:
        units = InterprocPipeline(cfg).build_units(path)
        results = [self.scanner.scan_unit(u) for u in units]   # + multiprocessing на Module 1
        return FindingAggregator(cfg).aggregate(results, units)
```

`ScanResult` — новые поля, **все с дефолтами** (иначе сломается
`scan_cli.py:172`, где идёт `ScanResult(**filtered)`):

```python
unit_id: str = ""
file: str = ""
function: str = ""
start_line: int = 0
end_line: int = 0
inline_depth_used: int = 0
inlined_functions: List[str] = field(default_factory=list)
provenance_files: List[str] = field(default_factory=list)
inline_truncated: bool = False
```

`ScannerConfig` — новые поля с префиксом `inline_*` / `project_*` (см. §8).
`scan_cli.py`: новые флаги `--project DIR`, `--inline-depth N`, `--inline-strategy`,
`--no-inline`, `--units-out`, `--project-report`, `--max-functions`, `--only-file`.
`scanner/report.py`: `format_project_report`, `to_json` для `ProjectScanResult`.

### 7.1. Дедупликация findings — обязательный шаг, которого нет в исходной идее

Если `g` инлайнена в `f`, то уязвимость в `g` будет отмечена **дважды**:
и при сканировании `g` как корня, и при сканировании `f`. На проекте из 5000 функций
это даёт лавину дублей и убивает precision в отчёте.

Правила агрегации (`FindingAggregator`):
1. Кластеризуем findings по пересечению множеств `provenance` (root + все inlined).
2. Внутри кластера побеждает **наиболее специфичный** юнит — минимальный по числу
   задействованных функций при сопоставимом (в пределах `epsilon`) hybrid_score.
   Интуиция: если и `g`, и `f+g` дают высокий скор, уязвимость в `g`.
3. Если корень `f` флагнут, а `g` (как самостоятельный корень) — нет,
   это сигнал «межпроцедурная уязвимость» → помечаем finding как `cross_function=true`.
   **Это ровно тот класс находок, ради которого делается модуль**, и его надо
   считать отдельной метрикой в экспериментах.
4. Стоимость: sandbox/LLM запускаем **только для победителей кластера**, а не для всех
   флагнутых юнитов (иначе LLM-бюджет умножается на коэффициент инлайна).

### 7.2. Стоимость project-level скана

Ориентир для проекта ~100k LOC / ~5000 функций:
- Module 1 (GNN, ~34 мс/юнит): ~3 мин на CPU, распараллеливается.
- Module 3–4 (LLM + Docker, секунды–минуты на юнит): **невозможно для всех**.
  Отсюда обязательны: порог детекции, дедуп (§7.1), `--max-findings`,
  сортировка по confidence и бюджет `--llm-budget N`.

---

## 8. Конфигурация

```python
@dataclass
class InliningConfig:
    enabled: bool = True
    max_depth: int = 2
    strategy: str = "priority"            # dfs | bfs | priority
    parser_backend: str = "auto"          # auto | treesitter | regex | joern

    # бюджеты
    max_tokens: int = 398                 # = block_size - 2, синхронизировать с ScannerConfig
    max_callee_tokens: int = 200
    max_inlined_nodes: int = 1000         # ограничение по узлам графа
    max_inline_sites: int = 32
    max_expansions_per_callee: int = 2
    bundle_max_tokens: int = 12000

    # семантика
    rename_on_collision: bool = True
    return_guard: str = "none"            # none | goto
    skip_labeled_callees: bool = True
    resolve_single_assign_fptr: bool = False
    allow_cross_tu_static: bool = False
    never_inline: list[str] = <DANGEROUS_SINKS + INPUT_SOURCES + libc/STL>
    debug_markers: bool = False

    # веса скоринга
    w_sink: float = 3.0; w_mem: float = 2.0; w_taint: float = 2.5
    w_small: float = 1.0; w_loop: float = 0.5
    w_depth: float = 1.0; w_trivial: float = 2.0

    # discovery
    include_globs: list[str] = ["**/*.c", "**/*.h", "**/*.cc", "**/*.cpp", ...]
    exclude_globs: list[str] = ["build/**", "third_party/**", ...]
    max_file_bytes: int = 2_000_000
    max_files: int = 20000
    include_dirs: list[str] = []
    cache_dir: str = ".interproc_cache"

    # roots
    root_min_tokens: int = 16             # порог «тривиальности», см. §14.5 —
                                          # применяется ТОЛЬКО к функциям без вызовов,
                                          # без sink'ов и без операций с памятью
    root_skip_test_files: bool = True
    dedupe_identical_bodies: bool = True
```

В `ScannerConfig` эти поля прорастают плоско (`inline_max_depth`, ...), плюс
опциональный `--config config.yaml` (пакет `yaml` в окружении есть), который
маппится и в `ScannerConfig`, и в `InliningConfig` — так закрывается пожелание
исходной идеи про YAML, без ломки текущего dataclass-стиля.

**`max_depth = 0` обязан давать `code_for_gnn`, побайтово равный исходному телу функции.**
Это регрессионный инвариант и первый тест, который надо написать.

---

## 9. Влияние на модель — риски и что с ними делать

Это самая недооценённая часть исходной идеи. «Архитектуру не трогаем» ≠ «качество не изменится».

| Риск | Механизм | Митигация |
|---|---|---|
| **Сдвиг распределения** | Модель обучена на одиночных функциях Devign; инлайненные юниты длиннее и структурно иные | (а) замерить D=0/1/2/3 на валидации; (б) LoRA-дообучение на инлайненных данных через готовый `code/run_finetune.py`; (в) перетюнить `beta` (`find_optimal_beta`) и `detection_threshold` **отдельно для каждой глубины** |
| **Потолок 398 токенов** | Инлайн почти всегда упирается в обрезку; при D=2 бóльшая часть тела корня может быть вытеснена | (а) метрика `truncation_rate` в статистике; (б) `strategy=priority`; (в) v2: sink-centric slicing — оставлять только строки на пути «параметр → sink», а не всё тело; (г) обсудить рост `block_size` (требует переобучения) |
| **Рост FP** | Больше кода = больше «опасных» паттернов на юнит | триаж-голова + дедуп (§7.1) + отдельный порог для инлайненного режима |
| **FAISS-соседи «поехали»** | Индекс построен по не-инлайненным эмбеддингам | опция пересборки индекса по инлайненным обучающим данным; замерить обе конфигурации |
| **Взрыв графа** | `preprocess_adj` паддит батч до max-длины и делает плотные матрицы `N×N` | ограничение `max_inlined_nodes` + мониторинг числа уникальных токенов на юнит |
| **Alpha-renaming ломает словарь** | новые токены `buf_i1` | переименование только при коллизии, короткие суффиксы |

Отдельно: перед экспериментами надо зафиксировать **baseline D=0 на project-level входе**
(разбить проект на функции без инлайна), иначе прирост от инлайна будет
неотличим от прироста от «мы вообще начали читать проект целиком».

---

## 10. Тестирование (модуль отдельно, до интеграции)

### 10.1. Синтетические мини-проекты `test_samples/projects/`

| Проект | Что проверяет |
|---|---|
| `p01_simple` | `main.c: handle()` → `util.c: copy_data()` со `strcpy`. Уязвимость видна **только** после инлайна. Мотивирующий пример для статьи |
| `p02_recursion` | прямая и взаимная рекурсия → skip("recursion"), отсутствие зависания |
| `p03_cpp_overload` | перегрузки и методы класса, `virtual` → skip |
| `p04_fptr` | вызов через указатель, callback |
| `p05_macro` | функциональный макрос, похожий на вызов; функция, определённая внутри макроса |
| `p06_static_dup` | две `static void helper()` с одним именем в разных файлах → правило TU |
| `p07_budget` | огромный callee → skip("budget"), корень не испорчен |
| `p08_ifdef` | `#ifdef`-ветвления, две платформенные реализации одной функции |
| `p09_headers` | определение в `.h`, `inline`-функции, include-цепочка глубиной 3 |

Для каждого — golden JSON с ожидаемыми рёбрами, инлайнами и причинами skip.

### 10.2. Property/инвариантные тесты
- `D=0` → `code_for_gnn == original_body` (побайтово).
- `tokens(D+1) >= tokens(D)` для одного и того же корня.
- `tokens_gnn <= max_tokens` **всегда** (никогда не превышаем бюджет).
- ни один callee не встречается чаще `max_expansions_per_callee`.
- детерминизм: два прогона на одном проекте дают идентичный список юнитов
  (сравнение по `unit_id` и хэшу текста).
- отсутствие бесконечных циклов на случайно сгенерированных call-графах с циклами.

### 10.3. Компилируемость
`code_for_sandbox` для всех мини-проектов проходит `gcc -fsyntax-only`
(тест `skipif` при отсутствии компилятора). Метрика по реальному репо — в отчёте stats.

### 10.4. Реальные проекты (smoke)
2–3 небольших OSS C-проекта. Замеряем: `parse_rate`, `resolution_rate`,
`inline_rate`, `avg_growth`, `truncation_rate`, `syntax_ok_rate`, время/функция, RSS.
Это же — таблица в статью.

### 10.5. Эффективность детекции (после интеграции)
Devign не подходит: там одиночные функции, межфайлового контекста нет by design.
Нужен бенчмарк с межпроцедурными случаями:
- **Juliet Test Suite (SARD)** — варианты `_1..._5x` и `bad`/`good` с source и sink,
  разнесёнными по функциям и файлам; это стандарт для оценки межпроцедурного анализа;
- дополнительно CVEFixes / D2A / реальные CVE с патчем, затрагивающим 2+ функции.

Метрики: Recall/Precision/ROC-AUC на подмножестве cross-function случаев,
отдельно от intra-function; ablation по `D ∈ {0,1,2,3}` × `strategy ∈ {dfs,priority}`
× `{с дообучением LoRA, без}`; стоимость (время, число LLM-вызовов, число sandbox-запусков).
Главный ожидаемый результат: **прирост recall на cross-function подмножестве
при неухудшении на intra-function.**

---

## 11. Этапы и критерии готовности

| # | Этап | Оценка | Критерий готовности |
|---|---|---|---|
| 0 | Скелет: `interproc/`, config, models, зависимости | 0.5 д | `import interproc` работает, конфиг сериализуется |
| 1 | Discovery + tree-sitter backend + regex fallback + кэш | 1.5 д | На p01–p09 найдены все функции; на реальном репо `parse_rate ≥ 95%` |
| 2 | Symbol table, include-граф, разрешение вызовов, SCC, `--emit-dot` | 1.5 д | Golden-тесты по рёбрам зелёные; `resolution_rate` в stats |
| 3 | Inliner: сегменты, source map, бюджеты, мемоизация, рекурсия, renaming | 2.5 д | Инвариантные тесты §10.2 зелёные |
| 4 | Scoring + стратегии dfs/bfs/priority | 1 д | Ablation-скрипт печатает stats по стратегиям |
| 5 | Bundler + проверка `-fsyntax-only` | 1 д | ≥80% юнитов мини-проектов компилируются |
| 6 | **Standalone CLI + JSONL юнитов + stats-отчёт** | 1 д | `python -m interproc.cli --project ./test_samples/projects/p01_simple --depth 2 --out units.jsonl` даёт юнит, где виден `strcpy` из другого файла ← **точка проверки, которую просил заказчик** |
| 7 | Интеграция: `scan_unit`, `ProjectScanner`, дедуп, ScannerConfig, CLI `--project`, отчёт | 2 д | `python scan_cli.py --project ... --dry-run` проходит end-to-end; старые 203 теста зелёные |
| 8 | Эксперименты: Juliet, ablations, LoRA-дообучение, калибровка порогов | 3+ д | Таблицы для статьи |

Между этапами 6 и 7 — контрольная точка с заказчиком, как и договаривались:
модуль проверяется изолированно на проектах, и только потом втыкается в ReGVD.

---

## 12. Открытые вопросы (нужно решение до этапа 1)

1. **Добавляем ли `tree-sitter` в зависимости проекта?** Альтернатива — жить на regex,
   но тогда качество разрешения вызовов и C++ сильно просядут. Рекомендация: добавить
   как *опциональную* (`pip install tree-sitter tree-sitter-c tree-sitter-cpp`),
   с работающим regex-fallback'ом, чтобы CI и соавторы не сломались.
2. **Только C или сразу C++?** Рекомендация: C — первый класс, C++ — best effort
   (перегрузки/шаблоны/виртуальные вызовы аккуратно пропускаем и считаем в stats).
3. **Дообучаем ли LoRA на инлайненных данных?** Влияет на объём эксперимента (§9).
   Рекомендация: да, но как отдельная ветка ablation, не как обязательное условие.
4. **Бенчмарк:** Juliet как основной — согласовать, т.к. от этого зависит вся глава
   экспериментов.
5. **Целевой масштаб проекта:** до 100k LOC или больше? От этого зависит,
   нужен ли инкрементальный режим (скан только изменённых файлов по git diff).

---

## 13. Формулировка для статьи (Module 0)

> **Module 0: Depth-Limited Interprocedural Inlining.**
> Given a project directory, the module builds a project-wide symbol table and call
> graph, and for each candidate function `F` constructs an *analysis unit*: an expanded
> representation in which the bodies of called functions are spliced into `F` up to a
> configurable depth `D`, subject to a tokenizer-aware budget matching the
> GraphCodeBERT input window. Call sites are inlined in order of a relevance score that
> favours callees containing dangerous sinks and callees receiving data derived from the
> caller's parameters, so that under a tight token budget the retained context is the
> security-relevant one. Each unit additionally carries a self-contained, compilable
> dependency bundle used by the exploit-adaptation and sandbox modules, and a source map
> that attributes every region of the expanded text back to its original file and line.
> The downstream modules (GraphCodeBERT, ReGCN + LoRA, FAISS, LLM adaptation, sandbox
> verification) are unchanged; interprocedural context is obtained entirely at the input
> representation level.

Плюс: признание limitation (потолок окна, unsound inlining, dynamic dispatch),
ablation по `D` и стратегиям, метрика cross-function recall.

---

## Приложение A. Полный список краевых случаев

**Парсинг / препроцессор**
- `#ifdef`/`#if 0` ветвления → парсим текст как есть; при несбалансированных
  директивах внутри тела ставим `unbalanced_preproc` и не инлайним такую функцию в другие.
- функции, порождённые макросами (`DEFINE_HANDLER(x) { ... }`) → не распознаются, в stats.
- функциональные макросы, выглядящие как вызовы → `resolution=macro`, skip.
- K&R-стиль объявлений параметров, `__attribute__`, `__declspec`, GNU-расширения.
- вложенные функции (GCC extension), лямбды C++ → не корни, не callee.
- `inline`/`static inline` в заголовках → определение в `.h`, доступное многим TU.
- CRLF, не-UTF8, минифицированный/сгенерированный код, файлы >2 МБ.
- symlink-петли, бинарные файлы, пустой проект.

**Разрешение вызовов**
- одноимённые `static` функции в разных файлах;
- перегрузки C++ по типам (арность совпадает) → ambiguous → skip;
- шаблоны → инлайним тело без подстановки типов (токены всё ещё информативны), помечаем;
- виртуальные методы, указатели на функции, callbacks, `dlsym` → skip;
- рекурсия прямая и взаимная, SCC;
- внешние библиотеки (объявление без определения) → external, остаётся вызовом;
- **sink-функции никогда не инлайним** (сохраняем токен-признак).

**Инлайнинг**
- ранний `return` в середине callee (поток управления ломается — осознанный компромисс);
- `goto`/метки → skip либо переименование меток;
- varargs → skip (нельзя связать параметры);
- `setjmp/longjmp`, inline asm → skip;
- коллизии имён локальных/параметров → переименование только при коллизии;
- один и тот же callee вызывается N раз → `max_expansions_per_callee`;
- callee больше корня → `max_callee_tokens`;
- исчерпание бюджета → вызов остаётся как есть, **частичных вставок нет**;
- корень без вызовов → юнит == исходная функция (полная обратная совместимость);
- корень сам по себе длиннее 398 токенов → инлайн бесполезен, WARNING + метрика.

**Пайплайн / отчёт**
- дубли findings «g» и «f, содержащая g» → кластеризация и выбор наиболее специфичного;
- атрибуция finding'а к `file:line` через source map (GNN даёт скор на всю функцию,
  не на строку — честно указываем диапазон корня + список provenance);
- стоимость LLM/sandbox на project-level → бюджеты и приоритизация;
- детерминизм и воспроизводимость (сортировки, хэши, версия конфига в `unit_id`).

---

## 14. Пошаговый процесс разбивки проекта на юниты (сквозной пример)

Здесь — точная последовательность стадий: что подаётся на вход каждой,
что она делает, что отдаёт наружу. Всё показано на одном сквозном примере.

### 14.0. Пример проекта

```
proj/
  include/util.h
  src/main.c
  src/util.c
```

`include/util.h`
```c
#ifndef UTIL_H
#define UTIL_H
#include <stddef.h>
void copy_data(char *dst, const char *src);
int  checksum(const char *s, size_t n);
#endif
```

`src/util.c`
```c
#include "util.h"
#include <string.h>

void copy_data(char *dst, const char *src) {
    strcpy(dst, src);
}

int checksum(const char *s, size_t n) {
    int acc = 0;
    for (size_t i = 0; i < n; i++) acc += s[i];
    return acc;
}
```

`src/main.c`
```c
#include "util.h"
#include <stdio.h>
#include <string.h>

static void log_line(const char *msg) {
    fprintf(stderr, "[log] %s\n", msg);
}

void handle_request(const char *user_input) {
    char buf[64];
    copy_data(buf, user_input);
    int c = checksum(buf, strlen(buf));
    printf("%d\n", c);
    log_line(buf);
}
```

Уязвимость (переполнение `buf[64]` через `strcpy`) **не видна ни в одном файле по
отдельности**: в `main.c` нет ни одного опасного вызова, в `util.c` нет буфера.
Это ровно тот случай, ради которого делается модуль.

---

### Стадия 1 — Discovery (обход проекта)

**Вход:** путь к корню проекта, `include_globs` / `exclude_globs`.
**Что делает:**
1. `os.walk(followlinks=False)`, накопление `realpath` посещённых каталогов (защита от symlink-петель).
2. Фильтр по расширениям и glob-исключениям (`build/`, `third_party/`, `.git/`, …).
3. Отсев: размер > `max_file_bytes`, наличие NUL в первых 8 КБ (бинарник).
4. Чтение как UTF-8 c `errors="replace"`, нормализация CRLF → LF
   (иначе поедут байтовые оффсеты и номера строк).
5. Проверка кэша по ключу `(path, size, mtime_ns, sha1[:16], parser_version)`.

**Выход:** `list[SourceFile]` — путь, относительный путь, текст, хэш, язык (`c`/`cpp`).

**Пример:** 3 файла — `include/util.h` (c), `src/util.c` (c), `src/main.c` (c).

---

### Стадия 2 — Parse (по каждому файлу независимо)

**Вход:** один `SourceFile`.
**Что делает:** tree-sitter парсит текст **как есть, без препроцессирования**
(поэтому не нужны ни include-пути, ни флаги компиляции), и из дерева извлекается:

| Что | Из каких узлов | Зачем |
|---|---|---|
| `FunctionDef` | `function_definition` | кандидаты в корни и в callee |
| `CallSite` | `call_expression` внутри тела функции | рёбра call-графа |
| `TypeDef` | `struct/union/enum_specifier`, `type_definition` | bundler |
| `GlobalDef` | `declaration` на верхнем уровне | bundler |
| `MacroDef` | `preproc_def`, `preproc_function_def` | отсечь «вызовы», которые на деле макросы |
| `Include` | `preproc_include` | include-граф + bundler |

Для каждой функции сразу считаются флаги: `has_goto`, `has_label`, `has_setjmp`,
`has_asm`, `has_preproc_branch`, `is_static`, `is_vararg`, `is_virtual`, `is_template`,
а также `body_hash` (для дедупа) и ленивый `token_count` (BPE).

Если tree-sitter недоступен — тот же контракт выполняет regex-backend
(на базе эвристик `analysis/taint.py:_analyze_heuristic`), с меньшей полнотой.

**Выход:** `FileFacts` на каждый файл (кэшируется на диск).

**Пример:**
```
src/util.c : copy_data(dst,src)[calls: strcpy], checksum(s,n)[calls: —]
src/main.c : log_line(msg)[static; calls: fprintf],
             handle_request(user_input)[calls: copy_data, checksum, strlen, printf, log_line]
include/util.h : функций нет, только прототипы + #include <stddef.h>
```

---

### Стадия 3 — Project Index (склейка фактов в общую картину)

**Вход:** все `FileFacts`.
**Что делает:**
1. **Таблица символов** `name -> [FunctionDef]` по всему проекту.
2. **Include-граф:** `#include "x.h"` резолвится относительно каталога файла, затем по
   `include_dirs`, затем по совпадению basename в проекте. `<x>` помечается системным.
   Считается транзитивное замыкание для каждого `.c` — это «область видимости» файла.
3. Таблицы типов, глобалов, макросов проекта.
4. Множество `external` — имена, у которых есть объявление, но нет определения в проекте.

**Выход:** `ProjectIndex`.

**Пример:** `src/main.c` включает `include/util.h`, значит `copy_data`/`checksum`
из `src/util.c` находятся в его области видимости.
`strcpy`, `printf`, `fprintf`, `strlen` → `external` + never-inline.

---

### Стадия 4 — Call Graph (разрешение вызовов)

**Вход:** `ProjectIndex` + все `CallSite`.
**Что делает:** для каждого call-site прогоняются правила §6.3 по порядку —
тот же файл → запрет на чужой `static` → приоритет include-замыкания →
фильтр по арности → единственный кандидат или `ambiguous` → never-inline список.
Затем `networkx` считает SCC (для защиты от циклов).

**Выход:** `CallGraph`: рёбра `(caller, callee|None, site, resolution, confidence)`.
Экспорт в DOT по `--emit-dot` для глазной проверки.

**Пример (рёбра из `handle_request`):**

| Вызов | Резолв | Причина |
|---|---|---|
| `copy_data(buf, user_input)` | → `src/util.c:copy_data:4` | `exact_include`, арность 2/2 |
| `checksum(buf, strlen(buf))` | → `src/util.c:checksum:8` | `exact_include`, арность 2/2 |
| `log_line(buf)` | → `src/main.c:log_line:5` | `exact_same_file` (static, тот же TU) |
| `strlen(...)` | ✗ | `external` + never_inline (sink/источник признаков) |
| `printf(...)` | ✗ | `external` + never_inline |

---

### Стадия 5 — Root Selection (какие функции становятся юнитами)

**Вход:** `ProjectIndex` + `CallGraph`.
**Что делает:** по умолчанию корнем становится **каждая** функция проекта. Отсеиваются:

1. функции из тестовых/сгенерированных файлов (`root_skip_test_files`);
2. дубликаты по `body_hash` (`dedupe_identical_bodies`) — остаётся одна, остальные
   помечаются как алиасы (важно для проектов с копипастой и для Juliet);
3. **тривиальные** функции. Важно: критерий не «короткая», а
   «нет вызовов **и** нет sink'ов **и** нет операций с памятью **и** `tokens < root_min_tokens`».
   Голый порог по длине выбросил бы `copy_data` — однострочник, который и есть уязвимость.

**Выход:** `list[FuncId]` — список корней.

**Пример:** 4 корня — `handle_request`, `copy_data`, `checksum`, `log_line`.
`copy_data` короткая, но содержит `strcpy` → остаётся корнем.

---

### Стадия 6 — Expansion (инлайнинг под бюджет) — по каждому корню

**Вход:** корень `F`, `CallGraph`, `InliningConfig`, токенный бюджет.
**Что делает:**

1. Собираются кандидаты — только рёбра с разрешённым callee.
2. Каждому считается score (§6.5): sink в теле, аргумент выведен из параметров корня,
   размер, глубина, тривиальность. Сортировка по убыванию, тай-брейк по `(file, line)`
   — детерминированно.
3. Проверки-отсечки по порядку: глубина → рекурсия (callee в текущем пути) →
   `max_expansions_per_callee` → `max_callee_tokens` → never-inline →
   опасные конструкции (`asm`, `setjmp`, varargs, метки).
4. Тело callee берётся из мемо-кэша `(fid, remaining_depth)` либо разворачивается рекурсивно.
5. Считается стоимость в **настоящих BPE-токенах**. Если не влезает целиком —
   вызов остаётся как есть (`skip: budget`), **частичных вставок нет**.
6. Вставка идёт **через сегменты по байтовым оффсетам**, не `str.replace`:
   текст режется на куски, каждый кусок помнит своё происхождение.

Что именно подставляется вместо `g(a, b)`:
```
[param_bind] T p1 = a; T p2 = b;
[body]       <тело g, при коллизии имён — суффикс _i1>
[ret_assign] return expr;  →  __r_g = expr;   (объявление __r_g ставится перед блоком,
                                               call-site заменяется на __r_g)
```

**Выход:** `code_for_gnn` + `segments` (source map) + `inlined[]` + `skipped[]` +
`tokens_gnn`, `truncated`, `depth_used`.

**Пример — `handle_request`, D=1:**

Порядок по score: `copy_data` (sink `strcpy` +3.0, аргумент из параметра +2.5, мелкая) →
`log_line` (sink `fprintf` +3.0, аргумент производный +2.5) →
`checksum` (sink'ов нет, +2.5 за аргумент).

Результат `code_for_gnn` (~120 BPE-токенов из 398 — влезло всё):
```c
void handle_request(const char *user_input) {
    char buf[64];
    { char *dst = buf; const char *src = user_input; strcpy(dst, src); }
    int __r_checksum;
    { const char *s = buf; size_t n = strlen(buf);
      int acc = 0; for (size_t i = 0; i < n; i++) acc += s[i];
      __r_checksum = acc; }
    int c = __r_checksum;
    printf("%d\n", c);
    { const char *msg = buf; fprintf(stderr, "[log] %s\n", msg); }
}
```

**Именно здесь и происходит то, ради чего всё затевалось:** `char buf[64]` и `strcpy`
впервые оказались в одном токенном окне, хотя лежат в разных файлах.
Коллизий имён нет (`dst/src/s/n/acc/i/msg` свободны) → переименования не было.
Комментариев-маркеров нет — они съедали бы бюджет и являются OOD-токенами.

`inlined`: `copy_data(d=1, +18 ток.)`, `log_line(d=1, +16)`, `checksum(d=1, +34)`.
`skipped`: `strlen(never_inline)`, `printf(never_inline)`.

---

### Стадия 7 — Bundle (компилируемое представление) — по каждому корню

**Вход:** тот же корень + множество достижимых callee.
**Что делает:** собирает самодостаточную единицу трансляции (§6.6):
системные includes → `#define` → типы в топологическом порядке → глобалы →
**полные определения callee (листья → корень)** → сама `F`.
Здесь **инлайна нет** — потому семантика точная и ранние `return` не проблема.

**Выход:** `code_for_sandbox`.

**Пример:**
```c
#include <stddef.h>
#include <stdio.h>
#include <string.h>

void copy_data(char *dst, const char *src) { strcpy(dst, src); }

int checksum(const char *s, size_t n) {
    int acc = 0;
    for (size_t i = 0; i < n; i++) acc += s[i];
    return acc;
}

static void log_line(const char *msg) { fprintf(stderr, "[log] %s\n", msg); }

void handle_request(const char *user_input) {
    char buf[64];
    copy_data(buf, user_input);
    int c = checksum(buf, strlen(buf));
    printf("%d\n", c);
    log_line(buf);
}
```
Проходит `gcc -fsyntax-only`. Именно это уходит в Module 3 (LLM строит harness,
вызывающий `handle_request` с длинной строкой) и в Module 4 (ASan ловит переполнение).

---

### Стадия 8 — Unit Assembly (сборка и выдача)

**Вход:** результаты стадий 6–7 по всем корням.
**Что делает:** формирует `AnalysisUnit`, считает `unit_id = sha1(root_fid + config_hash)`
(конфиг в ключе — чтобы результаты разных глубин не смешивались), собирает
статистику по проекту, сериализует в JSONL в формате, совместимом с текущим датасетом.

**Выход:** `units.jsonl` + `stats.json`.

**Пример — 4 юнита:**

| unit | корень | заинлайнено | токенов | что видно модели |
|---|---|---|---|---|
| u1 | `handle_request` | copy_data, checksum, log_line | ~120 | **buf[64] + strcpy вместе** |
| u2 | `copy_data` | — | ~18 | `strcpy` без контекста буфера |
| u3 | `checksum` | — | ~34 | ничего опасного |
| u4 | `log_line` | — | ~16 | format-функция с константной строкой |

---

### Стадия 9 — Передача в пайплайн (этап 7 дорожной карты)

```
для каждого юнита:
    Module 1  ← unit.code_for_gnn        (GNN + FAISS, hybrid_score)
    если hybrid_score >= threshold:
        Module 2  ← embedding             (retrieval эксплойтов)
        Module 3  ← unit.code_for_sandbox (LLM строит harness)
        Module 4  ← harness               (Docker + ASan/UBSan/MSan/TSan)
        Triage    ← unit.code_for_sandbox + inline-контекст
после всех юнитов:
    FindingAggregator: кластеризация по provenance, выбор наиболее специфичного
```

**Пример агрегации:** флагнуты `u1` (высокий скор) и `u2` (`strcpy` без контекста).
Их provenance пересекаются по `copy_data` → один кластер.
`u2` специфичнее (1 функция против 3), но у `u1` скор заметно выше и он даёт
компилируемый harness с реальным буфером → победитель `u1`,
finding помечается `cross_function=true`, `u2` идёт вложенным свидетельством.
В отчёт попадает **одна** находка:
`src/main.c:handle_request:9 → src/util.c:copy_data:4 (strcpy) — CWE-121, CONFIRMED`.

---

### 14.1. Сводка потока данных

```
проект/            Стадия 1  →  list[SourceFile]
                   Стадия 2  →  FileFacts (функции, вызовы, типы, глобалы, макросы)
                   Стадия 3  →  ProjectIndex (символы, include-граф)
                   Стадия 4  →  CallGraph (разрешённые рёбра)
                   Стадия 5  →  list[FuncId] (корни)
        для каждого корня:
                   Стадия 6  →  code_for_gnn + source map   (инлайн под бюджет)
                   Стадия 7  →  code_for_sandbox            (компилируемый bundle)
                   Стадия 8  →  AnalysisUnit → units.jsonl
                   Стадия 9  →  Module 1..4 → FindingAggregator → отчёт
```

Стадии 1–5 выполняются **один раз на проект**, стадии 6–8 — **по одному разу на функцию**
(с мемоизацией развёрнутых тел), стадия 9 — по одному разу на юнит,
причём дорогие Module 3–4 — только для победителей кластеров.
