import json
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


class VisualLabRunner:
    MIN_WIDTH = 320
    MAX_WIDTH = 1920

    MIN_HEIGHT = 240
    MAX_HEIGHT = 1080

    MIN_ITERATIONS = 64
    MAX_ITERATIONS = 5000

    MIN_ZOOM_MULTIPLIER = 1.001
    MAX_ZOOM_MULTIPLIER = 1.5

    MIN_INTERVAL_MS = 100
    MAX_INTERVAL_MS = 5000

    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.binary_dir = self.repo_root / "cuda_labs" / "mandelbrot"
        self.binary_path = self.binary_dir / "mandelbrot"

        self.output_dir = self.repo_root / "backend" / "generated" / "visual_lab"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.latest_image_path = self.output_dir / "latest.bmp"

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._status: dict[str, Any] = {
            "running": False,
            "message": "Idle",
            "frame_count": 0,
            "width": None,
            "height": None,
            "pixels": None,
            "max_iterations": None,
            "zoom_multiplier": None,
            "interval_ms": None,
            "current_zoom": None,
            "last_render_seconds": None,
            "last_result": None,
            "last_error": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            data = deepcopy(self._status)

        data["binary_exists"] = self.binary_path.exists()
        data["latest_image_exists"] = self.latest_image_path.exists()
        data["latest_image_url"] = "/cuda-visual/latest-image"

        return data

    def start(
        self,
        width: int,
        height: int,
        max_iterations: int,
        zoom_multiplier: float,
        interval_ms: int,
    ) -> dict[str, Any]:
        validation_error = self._validate_request(
            width=width,
            height=height,
            max_iterations=max_iterations,
            zoom_multiplier=zoom_multiplier,
            interval_ms=interval_ms,
        )

        if validation_error:
            return validation_error

        if not self.binary_path.exists():
            return {
                "running": False,
                "status": "build_required",
                "message": (
                    "mandelbrot binary was not found. Build it first with "
                    "`make` inside cuda_labs/mandelbrot."
                ),
                "binary_path": str(self.binary_path),
            }

        with self._lock:
            if self._status["running"]:
                data = deepcopy(self._status)
                data["status"] = "already_running"
                data["binary_exists"] = self.binary_path.exists()
                data["latest_image_exists"] = self.latest_image_path.exists()
                data["latest_image_url"] = "/cuda-visual/latest-image"
                return data

            self._status = {
                "running": True,
                "status": "running",
                "message": "Starting live CUDA visual generator.",
                "frame_count": 0,
                "width": width,
                "height": height,
                "pixels": width * height,
                "max_iterations": max_iterations,
                "zoom_multiplier": zoom_multiplier,
                "interval_ms": interval_ms,
                "current_zoom": 1.0,
                "last_render_seconds": None,
                "last_result": None,
                "last_error": None,
            }

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._render_loop,
            args=(width, height, max_iterations, zoom_multiplier, interval_ms),
            daemon=True,
        )
        self._thread.start()

        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

        with self._lock:
            self._status["running"] = False
            self._status["status"] = "stopped"
            if not self._status["last_error"]:
                self._status["message"] = "Live CUDA visual generator stopped."

        return self.status()

    def _render_loop(
        self,
        width: int,
        height: int,
        max_iterations: int,
        zoom_multiplier: float,
        interval_ms: int,
    ) -> None:
        """
        Render frames in the background.

        Basic strategy:
        - run the compiled CUDA binary once per frame
        - write a fresh BMP
        - swap it into the latest image path
        - increase zoom slightly
        - sleep briefly
        """
        frame_count = 0
        zoom = 1.0

        center_x = -0.5
        center_y = 0.0

        while not self._stop_event.is_set():
            temp_image_path = self.output_dir / "latest.tmp.bmp"

            command = [
                str(self.binary_path),
                str(temp_image_path),
                str(width),
                str(height),
                str(max_iterations),
                str(center_x),
                str(center_y),
                str(zoom),
            ]

            started_at = time.perf_counter()

            try:
                completed = subprocess.run(
                    command,
                    cwd=self.binary_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except Exception as exc:
                with self._lock:
                    self._status["running"] = False
                    self._status["status"] = "error"
                    self._status["message"] = "Failed to execute CUDA visual generator."
                    self._status["last_error"] = str(exc)
                return

            wall_seconds = round(time.perf_counter() - started_at, 4)

            if completed.returncode != 0:
                with self._lock:
                    self._status["running"] = False
                    self._status["status"] = "error"
                    self._status["message"] = "CUDA visual generator returned a non-zero exit code."
                    self._status["last_error"] = completed.stderr or completed.stdout
                return

            try:
                result = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                with self._lock:
                    self._status["running"] = False
                    self._status["status"] = "parse_error"
                    self._status["message"] = "Could not parse CUDA visual generator output."
                    self._status["last_error"] = str(exc)
                return

            if temp_image_path.exists():
                temp_image_path.replace(self.latest_image_path)

            frame_count += 1

            result["output_path"] = str(self.latest_image_path)

            with self._lock:
                self._status["running"] = True
                self._status["status"] = "running"
                self._status["message"] = "Live CUDA visual generator is running."
                self._status["frame_count"] = frame_count
                self._status["current_zoom"] = zoom
                self._status["last_render_seconds"] = wall_seconds
                self._status["last_result"] = result
                self._status["last_error"] = None

            zoom *= zoom_multiplier

            if self._stop_event.wait(interval_ms / 1000):
                break

        with self._lock:
            self._status["running"] = False
            self._status["status"] = "stopped"
            if not self._status["last_error"]:
                self._status["message"] = "Live CUDA visual generator stopped."

    def _validate_request(
        self,
        width: int,
        height: int,
        max_iterations: int,
        zoom_multiplier: float,
        interval_ms: int,
    ) -> dict[str, Any] | None:
        if width < self.MIN_WIDTH or width > self.MAX_WIDTH:
            return {
                "status": "invalid_request",
                "message": f"width must be between {self.MIN_WIDTH} and {self.MAX_WIDTH}.",
            }

        if height < self.MIN_HEIGHT or height > self.MAX_HEIGHT:
            return {
                "status": "invalid_request",
                "message": f"height must be between {self.MIN_HEIGHT} and {self.MAX_HEIGHT}.",
            }

        if max_iterations < self.MIN_ITERATIONS or max_iterations > self.MAX_ITERATIONS:
            return {
                "status": "invalid_request",
                "message": (
                    f"max_iterations must be between {self.MIN_ITERATIONS} "
                    f"and {self.MAX_ITERATIONS}."
                ),
            }

        if (
            zoom_multiplier < self.MIN_ZOOM_MULTIPLIER
            or zoom_multiplier > self.MAX_ZOOM_MULTIPLIER
        ):
            return {
                "status": "invalid_request",
                "message": (
                    f"zoom_multiplier must be between "
                    f"{self.MIN_ZOOM_MULTIPLIER} and {self.MAX_ZOOM_MULTIPLIER}."
                ),
            }

        if interval_ms < self.MIN_INTERVAL_MS or interval_ms > self.MAX_INTERVAL_MS:
            return {
                "status": "invalid_request",
                "message": (
                    f"interval_ms must be between "
                    f"{self.MIN_INTERVAL_MS} and {self.MAX_INTERVAL_MS}."
                ),
            }

        return None
