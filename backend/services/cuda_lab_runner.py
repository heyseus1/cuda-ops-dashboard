import json
import subprocess
import time
from pathlib import Path
from typing import Any


class CudaLabRunner:
    """
    CudaLabRunner owns execution of custom CUDA lab binaries.

    This class does not compile CUDA code.
    It safely runs already-built CUDA lab executables and parses their output.

    Current lab:
    - vector_add
    """

    MIN_ELEMENTS = 1_024
    MAX_ELEMENTS = 134_217_728

    MIN_THREADS_PER_BLOCK = 32
    MAX_THREADS_PER_BLOCK = 1024

    MIN_ITERATIONS = 1
    MAX_ITERATIONS = 10_000

    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.vector_add_dir = self.repo_root / "cuda_labs" / "vector_add"
        self.vector_add_binary = self.vector_add_dir / "vector_add"

    def get_vector_add_info(self) -> dict[str, Any]:
        return {
            "lab": "vector_add",
            "description": "Custom CUDA C++ vector addition kernel.",
            "binary_path": str(self.vector_add_binary),
            "binary_exists": self.vector_add_binary.exists(),
            "min_elements": self.MIN_ELEMENTS,
            "max_elements": self.MAX_ELEMENTS,
            "min_threads_per_block": self.MIN_THREADS_PER_BLOCK,
            "max_threads_per_block": self.MAX_THREADS_PER_BLOCK,
            "min_iterations": self.MIN_ITERATIONS,
            "max_iterations": self.MAX_ITERATIONS,
        }

    def run_vector_add(
        self,
        elements: int,
        threads_per_block: int,
        iterations: int,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        validation_error = self._validate_vector_add_request(
            elements=elements,
            threads_per_block=threads_per_block,
            iterations=iterations,
        )

        if validation_error:
            return validation_error

        if not self.vector_add_binary.exists():
            return {
                "status": "build_required",
                "passed": False,
                "message": (
                    "vector_add binary was not found. Build it first with "
                    "`make` inside cuda_labs/vector_add."
                ),
                "binary_path": str(self.vector_add_binary),
            }

        command = [
            str(self.vector_add_binary),
            str(elements),
            str(threads_per_block),
            str(iterations),
        ]

        start = time.perf_counter()

        try:
            completed = subprocess.run(
                command,
                cwd=self.vector_add_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )

            wall_seconds = round(time.perf_counter() - start, 4)

            parsed_output = None

            if completed.stdout.strip():
                parsed_output = json.loads(completed.stdout)

            return {
                "status": "completed" if completed.returncode == 0 else "failed",
                "passed": completed.returncode == 0,
                "return_code": completed.returncode,
                "wall_seconds": wall_seconds,
                "command": " ".join(command),
                "result": parsed_output,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "passed": False,
                "message": f"vector_add exceeded timeout of {timeout_seconds}s.",
                "command": " ".join(command),
            }

        except json.JSONDecodeError as exc:
            return {
                "status": "parse_error",
                "passed": False,
                "message": f"Could not parse vector_add JSON output: {exc}",
                "command": " ".join(command),
            }

        except Exception as exc:
            return {
                "status": "error",
                "passed": False,
                "message": str(exc),
                "command": " ".join(command),
            }

    def _validate_vector_add_request(
        self,
        elements: int,
        threads_per_block: int,
        iterations: int,
    ) -> dict[str, Any] | None:
        if elements < self.MIN_ELEMENTS or elements > self.MAX_ELEMENTS:
            return {
                "status": "invalid_request",
                "passed": False,
                "message": (
                    f"elements must be between "
                    f"{self.MIN_ELEMENTS} and {self.MAX_ELEMENTS}."
                ),
            }

        if (
            threads_per_block < self.MIN_THREADS_PER_BLOCK
            or threads_per_block > self.MAX_THREADS_PER_BLOCK
        ):
            return {
                "status": "invalid_request",
                "passed": False,
                "message": (
                    f"threads_per_block must be between "
                    f"{self.MIN_THREADS_PER_BLOCK} and "
                    f"{self.MAX_THREADS_PER_BLOCK}."
                ),
            }

        if iterations < self.MIN_ITERATIONS or iterations > self.MAX_ITERATIONS:
            return {
                "status": "invalid_request",
                "passed": False,
                "message": (
                    f"iterations must be between "
                    f"{self.MIN_ITERATIONS} and {self.MAX_ITERATIONS}."
                ),
            }

        return None
