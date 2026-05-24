"""
Sandbox Executor — runs exploit harnesses in Docker containers with strict isolation.

Inspired by VBF (Gajjar 2026):
  - Docker containers with resource limits
  - ASan/UBSan sanitizers for C/C++
  - Valgrind as secondary memory checker
  - Evidence collection from stdout/stderr/exit code
  - Asymmetric validation: more payloads for flag=1, fewer for flag=0
"""

import os
import re
import time
import logging
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Optional, List

from .evidence import EvidenceClassifier, EvidenceLevel, ClassificationResult
from . import docker_manager

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 10_000

SANITIZER_PROFILES = {
    "asan_ubsan": {
        "compile_gcc": "gcc -fsanitize=address,undefined -g -fno-omit-frame-pointer",
        "compile_clang": "clang -fsanitize=address,undefined -g -fno-omit-frame-pointer",
        "compile_fuzz": "clang -fsanitize=fuzzer,address,undefined -g -fno-omit-frame-pointer",
        "env": {
            "ASAN_OPTIONS": "detect_leaks=0:print_stats=0:halt_on_error=0",
            "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=0",
        },
        "detects": [
            "buffer_overflow", "stack_buffer_overflow", "heap_buffer_overflow",
            "use_after_free", "double_free", "null_pointer_deref",
            "format_string", "off_by_one",
        ],
    },
    "msan": {
        "compile_gcc": None,
        "compile_clang": "clang -fsanitize=memory -g -fno-omit-frame-pointer -fno-optimize-sibling-calls",
        "compile_fuzz": "clang -fsanitize=fuzzer,memory -g -fno-omit-frame-pointer",
        "env": {
            "MSAN_OPTIONS": "print_stats=0:halt_on_error=0",
        },
        "detects": ["uninitialized_read", "uninitialized_variable"],
    },
    "tsan": {
        "compile_gcc": None,
        "compile_clang": "clang -fsanitize=thread -g -lpthread",
        "compile_fuzz": "clang -fsanitize=fuzzer,thread -g -lpthread",
        "env": {
            "TSAN_OPTIONS": "print_stats=0:halt_on_error=0",
        },
        "detects": ["race_condition", "deadlock", "toctou"],
    },
}


@dataclass
class SandboxConfig:
    memory_limit: str = "1g"
    cpu_quota: float = 0.9
    network: str = "none"
    read_only: bool = True
    tmpfs_size: str = "256m"
    pids_limit: int = 128
    timeout: int = 30
    docker_image: str = ""
    security_opt: str = "no-new-privileges"
    use_seccomp: bool = True
    use_valgrind: bool = False
    cap_drop: str = "ALL"
    use_multi_sanitizer: bool = False
    use_libfuzzer: bool = False
    fuzz_time: int = 15
    prefer_clang: bool = False


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    evidence: ClassificationResult
    execution_time: float = 0.0
    timed_out: bool = False
    harness_compiled: bool = False
    compilation_error: str = ""
    valgrind_output: str = ""
    sanitizer_profile: str = ""
    fuzz_mode: bool = False


class SandboxExecutor:
    """
    Executes exploit harnesses in isolated Docker containers.

    Flow:
      1. Write harness code to temp directory
      2. Launch Docker container with strict limits
      3. Compile harness with sanitizers inside container
      4. Execute and capture output
      5. (Optional) Run Valgrind for additional checks
      6. Classify evidence
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.classifier = EvidenceClassifier()
        self._docker_available = None
        self._image_ready = False

    @property
    def docker_available(self) -> bool:
        if self._docker_available is None:
            try:
                subprocess.run(["docker", "info"], capture_output=True,
                               timeout=10, check=True)
                self._docker_available = True
            except (subprocess.CalledProcessError, FileNotFoundError,
                    subprocess.TimeoutExpired):
                self._docker_available = False
        return self._docker_available

    def ensure_image(self) -> bool:
        if self._image_ready:
            return True
        if not self.docker_available:
            return False
        image = self.config.docker_image or docker_manager.get_image_name()
        if docker_manager.image_exists(image.split(":")[-1] if ":" in image else "c-cpp"):
            self._image_ready = True
            return True
        if docker_manager.build_image():
            self.config.docker_image = docker_manager.get_image_name()
            self._image_ready = True
            return True
        if not self.config.docker_image:
            self.config.docker_image = "gcc:14"
        return True

    def execute(self, harness_code: str, payload: str,
                compilation_cmd: str,
                expected_signals: list = None,
                timeout: Optional[int] = None) -> ExecutionResult:
        timeout = timeout or self.config.timeout

        if self.docker_available:
            self.ensure_image()
            return self._execute_docker(harness_code, payload, compilation_cmd,
                                        expected_signals, timeout)
        return self._execute_local(harness_code, payload, compilation_cmd,
                                   expected_signals, timeout)

    def execute_batch(self, harnesses: list,
                      expected_signals_list: list = None,
                      early_stop: bool = True) -> List[ExecutionResult]:
        """
        Execute multiple harnesses, optionally stopping on first CONFIRMED.

        Args:
            harnesses: list of dicts {harness_code, payload, compilation_cmd}
            expected_signals_list: parallel list of expected_signals per harness
            early_stop: stop after first CONFIRMED result

        Returns:
            list of ExecutionResult
        """
        results = []
        if expected_signals_list is None:
            expected_signals_list = [None] * len(harnesses)

        for i, harness in enumerate(harnesses):
            result = self.execute(
                harness_code=harness["harness_code"],
                payload=harness["payload"],
                compilation_cmd=harness["compilation_cmd"],
                expected_signals=expected_signals_list[i],
            )
            results.append(result)

            if early_stop and result.evidence.level == EvidenceLevel.CONFIRMED:
                logger.info(f"Early stop: CONFIRMED at harness #{i+1}")
                break

        return results

    def execute_multi_sanitizer(self, harness_code: str, payload: str,
                                vuln_type: str = "",
                                expected_signals: list = None,
                                timeout: Optional[int] = None) -> List[ExecutionResult]:
        """
        Run harness with multiple sanitizer profiles.

        Selects relevant profiles based on vuln_type, runs static then fuzz
        for each. Returns all results, best CONFIRMED first.
        """
        timeout = timeout or self.config.timeout
        profiles = self._select_profiles(vuln_type)
        results = []

        for profile_name, profile in profiles:
            compile_cmd = self._get_compile_cmd(profile)
            if not compile_cmd:
                continue

            result = self._execute_local(
                harness_code, payload, compile_cmd,
                expected_signals, timeout,
            )
            result.sanitizer_profile = profile_name
            results.append(result)

            if result.evidence.level == EvidenceLevel.CONFIRMED:
                return results

            if self.config.use_libfuzzer and self._has_clang():
                fuzz_result = self._execute_fuzz_local(
                    harness_code, profile, expected_signals,
                )
                if fuzz_result:
                    fuzz_result.sanitizer_profile = profile_name
                    fuzz_result.fuzz_mode = True
                    results.append(fuzz_result)
                    if fuzz_result.evidence.level == EvidenceLevel.CONFIRMED:
                        return results

        return results

    def _select_profiles(self, vuln_type: str) -> list:
        """Select sanitizer profiles relevant for the given vuln_type."""
        if not vuln_type or not self.config.use_multi_sanitizer:
            return [("asan_ubsan", SANITIZER_PROFILES["asan_ubsan"])]

        relevant = []
        for name, profile in SANITIZER_PROFILES.items():
            if vuln_type in profile["detects"]:
                relevant.append((name, profile))

        if not relevant:
            relevant = [("asan_ubsan", SANITIZER_PROFILES["asan_ubsan"])]
        return relevant

    def _get_compile_cmd(self, profile: dict) -> Optional[str]:
        """Get compilation command, preferring clang if available."""
        if self.config.prefer_clang or profile["compile_gcc"] is None:
            if self._has_clang():
                return profile["compile_clang"]
            if profile["compile_gcc"] is None:
                return None
        return profile["compile_gcc"] or profile.get("compile_clang")

    def _execute_fuzz_local(self, harness_code: str, profile: dict,
                            expected_signals: list = None) -> Optional[ExecutionResult]:
        """
        Run harness with LibFuzzer. Wraps code with LLVMFuzzerTestOneInput
        if not already present, compiles with -fsanitize=fuzzer, runs for
        config.fuzz_time seconds.
        """
        if not self._has_clang():
            return None

        fuzz_compile = profile.get("compile_fuzz")
        if not fuzz_compile:
            return None

        if "LLVMFuzzerTestOneInput" not in harness_code:
            fuzz_code = self._wrap_as_fuzz_harness(harness_code)
            if not fuzz_code:
                return None
        else:
            fuzz_code = harness_code

        workdir = tempfile.mkdtemp(prefix="sandbox_fuzz_")
        try:
            harness_path = os.path.join(workdir, "test_fuzz.c")
            with open(harness_path, "w") as f:
                f.write(fuzz_code)

            binary_path = os.path.join(workdir, "test_fuzz")
            corpus_dir = os.path.join(workdir, "corpus")
            os.makedirs(corpus_dir)

            full_compile = f"{fuzz_compile} {harness_path} -o {binary_path}"
            compile_result = subprocess.run(
                full_compile, shell=True, capture_output=True, text=True,
                timeout=30, cwd=workdir,
            )
            if compile_result.returncode != 0:
                logger.debug(f"Fuzz compilation failed: {compile_result.stderr[:300]}")
                return None

            env = {**os.environ, **profile.get("env", {})}
            fuzz_time = self.config.fuzz_time

            start = time.time()
            timed_out = False
            try:
                run_result = subprocess.run(
                    [binary_path, corpus_dir,
                     f"-max_total_time={fuzz_time}",
                     "-print_final_stats=1",
                     f"-jobs=1", "-workers=1"],
                    capture_output=True, text=True,
                    timeout=fuzz_time + 30, cwd=workdir, env=env,
                )
                elapsed = time.time() - start
                exit_code = run_result.returncode
                stdout = run_result.stdout or ""
                stderr = run_result.stderr or ""
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                timed_out = True
                exit_code, stdout, stderr = -1, "", "LibFuzzer timeout"

            evidence = self.classifier.classify(
                exit_code=exit_code, stdout=stdout, stderr=stderr,
                timed_out=timed_out, expected_signals=expected_signals,
            )

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout[:MAX_OUTPUT_BYTES],
                stderr=stderr[:MAX_OUTPUT_BYTES],
                evidence=evidence,
                execution_time=elapsed,
                timed_out=timed_out,
                harness_compiled=True,
                fuzz_mode=True,
                sanitizer_profile="libfuzzer",
            )
        except Exception as e:
            logger.warning(f"Fuzz execution error: {e}")
            return None
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _wrap_as_fuzz_harness(harness_code: str) -> Optional[str]:
        """
        Transform a static main()-based harness into a LibFuzzer harness.

        Extracts the function being tested and wraps it with
        LLVMFuzzerTestOneInput. Returns None if transformation fails.
        """
        if "int main" not in harness_code:
            return None

        func_match = re.search(
            r'(?:void|int|char|long|unsigned|short|float|double|size_t)\s*\*?\s+'
            r'(\w+)\s*\([^)]*\)\s*\{',
            harness_code,
        )
        func_name = func_match.group(1) if func_match else None
        if func_name in ("main", None):
            pattern = re.compile(
                r'(?:(?:static|void|int|char|long|unsigned|short|float|double|'
                r'size_t|ssize_t)\s*\*?\s+)+(\w+)\s*\(',
            )
            skip = {"if", "for", "while", "switch", "return", "sizeof", "main",
                    "printf", "sprintf", "fprintf", "snprintf", "memset",
                    "memcpy", "strcpy", "strncpy", "malloc", "free", "calloc"}
            for m in pattern.finditer(harness_code):
                name = m.group(1)
                if name not in skip:
                    func_name = name
                    break

        if not func_name or func_name == "main":
            return None

        main_pattern = re.compile(
            r'int\s+main\s*\([^)]*\)\s*\{.*\}',
            re.DOTALL,
        )
        code_without_main = main_pattern.sub("", harness_code)

        fuzz_entry = f"""
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {{
    if (size == 0) return 0;
    char *input = (char *)malloc(size + 1);
    if (!input) return 0;
    memcpy(input, data, size);
    input[size] = '\\0';
    {func_name}(input);
    free(input);
    return 0;
}}
"""
        if "#include <stdint.h>" not in code_without_main:
            code_without_main = "#include <stdint.h>\n" + code_without_main
        if "#include <stddef.h>" not in code_without_main:
            code_without_main = "#include <stddef.h>\n" + code_without_main

        return code_without_main.rstrip() + "\n" + fuzz_entry

    @staticmethod
    def _has_clang() -> bool:
        try:
            subprocess.run(["clang", "--version"],
                           capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _execute_docker(self, harness_code: str, payload: str,
                        compilation_cmd: str, expected_signals: list,
                        timeout: int) -> ExecutionResult:
        workdir = tempfile.mkdtemp(prefix="sandbox_")
        try:
            self._write_harness_files(workdir, harness_code, payload)

            script = self._build_run_script(compilation_cmd, timeout,
                                            self.config.use_valgrind)
            script_path = os.path.join(workdir, "run.sh")
            with open(script_path, "w") as f:
                f.write(script)

            docker_cmd = self._build_docker_cmd(workdir, timeout)

            start = time.time()
            timed_out = False
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True, text=True,
                    timeout=timeout + 45,
                )
                elapsed = time.time() - start
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                timed_out = True
                result = _empty_result("Docker timeout")

            return self._parse_execution_output(
                result, elapsed, timed_out, expected_signals
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _execute_local(self, harness_code: str, payload: str,
                       compilation_cmd: str, expected_signals: list,
                       timeout: int) -> ExecutionResult:
        workdir = tempfile.mkdtemp(prefix="sandbox_local_")
        try:
            harness_path = os.path.join(workdir, "test_harness.c")
            with open(harness_path, "w") as f:
                f.write(harness_code)

            binary_path = os.path.join(workdir, "test_harness")
            full_compile = f"{compilation_cmd} {harness_path} -o {binary_path}"

            compile_result = subprocess.run(
                full_compile, shell=True, capture_output=True, text=True,
                timeout=30, cwd=workdir
            )

            if compile_result.returncode != 0:
                return ExecutionResult(
                    exit_code=-1, stdout="", stderr=compile_result.stderr,
                    evidence=self.classifier.classify(-1, "", compile_result.stderr),
                    harness_compiled=False,
                    compilation_error=compile_result.stderr[:MAX_OUTPUT_BYTES],
                )

            sanitizer_env = {
                **os.environ,
                "ASAN_OPTIONS": "detect_leaks=0:print_stats=0:halt_on_error=0",
                "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=0",
                "MALLOC_CHECK_": "3",
            }

            start = time.time()
            timed_out = False
            try:
                run_result = subprocess.run(
                    [binary_path], input=payload,
                    capture_output=True, text=True,
                    timeout=timeout, cwd=workdir, env=sanitizer_env,
                )
                elapsed = time.time() - start
                exit_code = run_result.returncode
                stdout, stderr = run_result.stdout, run_result.stderr
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                timed_out = True
                exit_code, stdout, stderr = -1, "", "Timeout"

            valgrind_output = ""
            if self.config.use_valgrind and not timed_out and self._has_valgrind():
                valgrind_output = self._run_valgrind_local(
                    binary_path, payload, timeout, workdir
                )
                stderr = stderr + "\n" + valgrind_output

            evidence = self.classifier.classify(
                exit_code=exit_code, stdout=stdout, stderr=stderr,
                timed_out=timed_out, expected_signals=expected_signals,
            )

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout[:MAX_OUTPUT_BYTES],
                stderr=stderr[:MAX_OUTPUT_BYTES],
                evidence=evidence,
                execution_time=elapsed,
                timed_out=timed_out,
                harness_compiled=True,
                valgrind_output=valgrind_output[:MAX_OUTPUT_BYTES],
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _write_harness_files(self, workdir: str, harness_code: str, payload: str):
        with open(os.path.join(workdir, "test_harness.c"), "w") as f:
            f.write(harness_code)
        with open(os.path.join(workdir, "payload.txt"), "w") as f:
            f.write(payload)

    def _build_run_script(self, compilation_cmd: str, timeout: int,
                          use_valgrind: bool) -> str:
        lines = [
            "#!/bin/bash",
            "set -e",
            f"cd /work",
            f"{compilation_cmd} test_harness.c -o /out/test_harness 2>&1",
            "echo '===COMPILATION_OK==='",
            "set +e",
            f"cat payload.txt | timeout {timeout} /out/test_harness 2>&1",
            "ASAN_EXIT=$?",
            'echo "EXIT_CODE=$ASAN_EXIT"',
        ]
        if use_valgrind:
            lines.extend([
                "echo '===VALGRIND_START==='",
                f"cat payload.txt | timeout {timeout} valgrind "
                "--leak-check=full --track-origins=yes --error-exitcode=42 "
                "/out/test_harness 2>&1",
                "echo '===VALGRIND_END==='",
            ])
        return "\n".join(lines) + "\n"

    def _build_docker_cmd(self, workdir: str, timeout: int) -> list:
        image = self.config.docker_image or docker_manager.get_image_name()

        cmd = [
            "docker", "run", "--rm",
            f"--memory={self.config.memory_limit}",
            f"--cpus={self.config.cpu_quota}",
            f"--network={self.config.network}",
            f"--pids-limit={self.config.pids_limit}",
            f"--security-opt={self.config.security_opt}",
            f"--cap-drop={self.config.cap_drop}",
            "--tmpfs", f"/tmp:size={self.config.tmpfs_size},noexec,nosuid",
            "--tmpfs", "/out:size=64m,exec",
            "-v", f"{workdir}:/work:ro",
            "--workdir", "/work",
            "-e", "ASAN_OPTIONS=detect_leaks=0:print_stats=0:halt_on_error=0",
            "-e", "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0",
            "-e", "MALLOC_CHECK_=3",
            "--label", "gnn-regvd-sandbox",
        ]

        if self.config.use_seccomp:
            seccomp_path = docker_manager.get_seccomp_profile_path()
            if os.path.exists(seccomp_path):
                cmd.extend(["--security-opt", f"seccomp={seccomp_path}"])

        cmd.extend([image, "bash /work/run.sh"])
        return cmd

    def _parse_execution_output(self, result, elapsed: float,
                                timed_out: bool,
                                expected_signals: list) -> ExecutionResult:
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        compiled = "===COMPILATION_OK===" in output

        exit_code = result.returncode
        m = re.search(r'EXIT_CODE=(\d+)', output)
        if m:
            exit_code = int(m.group(1))

        valgrind_output = ""
        valgrind_match = re.search(
            r'===VALGRIND_START===(.*?)===VALGRIND_END===', output, re.DOTALL
        )
        if valgrind_match:
            valgrind_output = valgrind_match.group(1).strip()

        full_stderr = (result.stderr or "")
        if valgrind_output:
            full_stderr = full_stderr + "\n" + valgrind_output

        evidence = self.classifier.classify(
            exit_code=exit_code,
            stdout=result.stdout or "",
            stderr=full_stderr,
            timed_out=timed_out,
            expected_signals=expected_signals,
        )

        compilation_error = ""
        if not compiled:
            compilation_error = output[:MAX_OUTPUT_BYTES]

        return ExecutionResult(
            exit_code=exit_code,
            stdout=(result.stdout or "")[:MAX_OUTPUT_BYTES],
            stderr=(result.stderr or "")[:MAX_OUTPUT_BYTES],
            evidence=evidence,
            execution_time=elapsed,
            timed_out=timed_out,
            harness_compiled=compiled,
            compilation_error=compilation_error,
            valgrind_output=valgrind_output[:MAX_OUTPUT_BYTES],
        )

    def _run_valgrind_local(self, binary_path: str, payload: str,
                            timeout: int, cwd: str) -> str:
        try:
            result = subprocess.run(
                ["valgrind", "--leak-check=full", "--track-origins=yes",
                 "--error-exitcode=42", binary_path],
                input=payload, capture_output=True, text=True,
                timeout=timeout, cwd=cwd,
            )
            return result.stderr[:MAX_OUTPUT_BYTES]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    @staticmethod
    def _has_valgrind() -> bool:
        try:
            subprocess.run(["valgrind", "--version"],
                           capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


def _empty_result(stderr_msg: str):
    return type('Result', (), {
        'stdout': '', 'stderr': stderr_msg, 'returncode': -1
    })()
