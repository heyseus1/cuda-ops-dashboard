from datetime import datetime, timezone
from typing import Any

import threading
import time

import torch


class BenchmarkRunner:
    """
    BenchmarkRunner owns the CUDA workload lifecycle.

    This class is responsible for:
    - Starting benchmark jobs
    - Stopping benchmark jobs safely
    - Tracking benchmark status
    - Tracking benchmark logs
    - Applying safety limits
    """

    MAX_MATRIX_SIZE = 16000
    MIN_MATRIX_SIZE = 512

    MAX_ITERATIONS = 250
    MIN_ITERATIONS = 1

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.state: dict[str, Any] = {
            "status": "idle",
            "running": False,
            "stop_requested": False,
            "last_seconds": None,
            "total_seconds": None,
            "last_size": None,
            "total_iterations": None,
            "completed_iterations": 0,
            "last_error": None,
            "started_at": None,
            "finished_at": None,
            "logs": [],
        }

    def start(self, size: int, iterations: int) -> dict[str, Any]:
        """
        Start a benchmark in a background thread.
        """

        validation_error = self._validate_request(size=size, iterations=iterations)

        if validation_error:
            return validation_error

        with self.lock:
            if self.state["running"]:
                return {
                    "status": "already_running",
                    "message": "A benchmark is already running.",
                    "benchmark": self._snapshot_unlocked(),
                }

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            args=(size, iterations),
            daemon=True,
        )

        self.thread.start()

        return {
            "status": "started",
            "size": size,
            "iterations": iterations,
            "max_matrix_size": self.MAX_MATRIX_SIZE,
            "max_iterations": self.MAX_ITERATIONS,
        }

    def stop(self) -> dict[str, str]:
        """
        Request benchmark stop.

        This is cooperative cancellation. It stops between CUDA iterations.
        It does not always kill an in-progress CUDA kernel instantly.
        """

        with self.lock:
            if not self.state["running"]:
                return {
                    "status": "not_running",
                    "message": "No benchmark is currently running.",
                }

            self.state["status"] = "stopping"
            self.state["stop_requested"] = True

        self.stop_event.set()
        self._add_log("Stop requested from UI/API.")

        return {
            "status": "stop_requested",
            "message": "Benchmark will stop safely after the current CUDA operation finishes.",
        }

    def get_status(self) -> dict[str, Any]:
        """
        Return a safe copy of the benchmark state.
        """

        with self.lock:
            return self._snapshot_unlocked()

    def _run(self, size: int, iterations: int) -> None:
        """
        Execute the actual CUDA workload.
        """

        start_total = time.time()

        self._reset_state_for_run(size=size, iterations=iterations)
        self._add_log(f"Benchmark started: size={size}, iterations={iterations}")

        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system.")

            for iteration in range(1, iterations + 1):
                if self.stop_event.is_set():
                    self._mark_stopped()
                    self._add_log(
                        "Stop requested. Benchmark stopped safely between iterations."
                    )
                    break

                self._add_log(f"Starting iteration {iteration}/{iterations}")

                x = torch.randn(size, size, device="cuda")
                y = torch.randn(size, size, device="cuda")

                start_iteration = time.time()

                _ = x @ y

                torch.cuda.synchronize()

                iteration_seconds = round(time.time() - start_iteration, 4)

                del x
                del y

                with self.lock:
                    self.state["last_seconds"] = iteration_seconds
                    self.state["completed_iterations"] = iteration

                self._add_log(
                    f"Finished iteration {iteration}/{iterations} in {iteration_seconds}s"
                )

            else:
                self._mark_completed()
                self._add_log("Benchmark completed successfully.")

        except torch.cuda.OutOfMemoryError as exc:
            self._mark_error(f"CUDA out of memory: {exc}")
            self._add_log("CUDA out-of-memory error. Try reducing matrix size.")

        except Exception as exc:
            self._mark_error(str(exc))
            self._add_log(f"Benchmark failed: {exc}")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with self.lock:
                self.state["running"] = False
                self.state["total_seconds"] = round(time.time() - start_total, 4)
                self.state["finished_at"] = self._utc_now()

                if self.state["status"] == "running":
                    self.state["status"] = "idle"

            self.stop_event.clear()
            self._add_log("GPU cache cleared. Benchmark worker finished.")

    def _validate_request(self, size: int, iterations: int) -> dict[str, Any] | None:
        """
        Validate benchmark input before starting a workload.
        """

        if size < self.MIN_MATRIX_SIZE or size > self.MAX_MATRIX_SIZE:
            return {
                "status": "invalid_request",
                "message": (
                    f"Matrix size must be between "
                    f"{self.MIN_MATRIX_SIZE} and {self.MAX_MATRIX_SIZE}."
                ),
            }

        if iterations < self.MIN_ITERATIONS or iterations > self.MAX_ITERATIONS:
            return {
                "status": "invalid_request",
                "message": (
                    f"Iterations must be between "
                    f"{self.MIN_ITERATIONS} and {self.MAX_ITERATIONS}."
                ),
            }

        return None

    def _reset_state_for_run(self, size: int, iterations: int) -> None:
        """
        Reset benchmark state at the start of a new benchmark.
        """

        with self.lock:
            self.state["status"] = "running"
            self.state["running"] = True
            self.state["stop_requested"] = False
            self.state["last_seconds"] = None
            self.state["total_seconds"] = None
            self.state["last_size"] = size
            self.state["total_iterations"] = iterations
            self.state["completed_iterations"] = 0
            self.state["last_error"] = None
            self.state["started_at"] = self._utc_now()
            self.state["finished_at"] = None
            self.state["logs"] = []

    def _mark_completed(self) -> None:
        with self.lock:
            self.state["status"] = "completed"

    def _mark_stopped(self) -> None:
        with self.lock:
            self.state["status"] = "stopped"
            self.state["stop_requested"] = True

    def _mark_error(self, error_message: str) -> None:
        with self.lock:
            self.state["status"] = "error"
            self.state["last_error"] = error_message

    def _add_log(self, message: str) -> None:
        """
        Add one console-style log line to the benchmark state.
        """

        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"

        with self.lock:
            self.state["logs"].append(line)
            self.state["logs"] = self.state["logs"][-50:]

    def _snapshot_unlocked(self) -> dict[str, Any]:
        """
        Copy benchmark state.

        This should only be called while self.lock is already held.
        """

        snapshot = dict(self.state)
        snapshot["logs"] = list(self.state["logs"])
        return snapshot

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
