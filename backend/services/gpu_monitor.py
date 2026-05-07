from pathlib import Path
from typing import Any

import pynvml
import torch


class GpuMonitor:
    """
    GpuMonitor owns all NVIDIA Management Library / NVML interactions.
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def get_metrics(self) -> dict[str, Any]:
        """
        Return live GPU metrics used by the dashboard home page.
        """

        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

        temperature = pynvml.nvmlDeviceGetTemperature(
            self.handle,
            pynvml.NVML_TEMPERATURE_GPU,
        )

        power_watts = round(
            pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000,
            2,
        )

        power_limit_watts = self._safe_nvml_call(
            lambda: round(
                pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000,
                2,
            )
        )

        power_percent = None
        power_headroom_watts = None

        if power_limit_watts:
            power_percent = round((power_watts / power_limit_watts) * 100, 2)
            power_headroom_watts = round(power_limit_watts - power_watts, 2)

        graphics_clock_mhz = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                self.handle,
                pynvml.NVML_CLOCK_GRAPHICS,
            )
        )

        sm_clock_mhz = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                self.handle,
                pynvml.NVML_CLOCK_SM,
            )
        )

        memory_clock_mhz = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                self.handle,
                pynvml.NVML_CLOCK_MEM,
            )
        )

        performance_state = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetPerformanceState(self.handle)
        )

        memory_used_mb = self._bytes_to_mb(memory.used)
        memory_total_mb = self._bytes_to_mb(memory.total)
        memory_free_mb = self._bytes_to_mb(memory.free)

        memory_percent = 0

        if memory.total:
            memory_percent = round((memory.used / memory.total) * 100, 2)

        process_count = self.get_processes()["count"]

        return {
            "gpu": self.get_gpu_name(),
            "cuda_available": torch.cuda.is_available(),
            "utilization_percent": utilization.gpu,
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "memory_free_mb": memory_free_mb,
            "memory_percent": memory_percent,
            "temperature_c": temperature,
            "power_watts": power_watts,
            "power_limit_watts": power_limit_watts,
            "power_percent": power_percent,
            "power_headroom_watts": power_headroom_watts,
            "graphics_clock_mhz": graphics_clock_mhz,
            "sm_clock_mhz": sm_clock_mhz,
            "memory_clock_mhz": memory_clock_mhz,
            "performance_state": performance_state,
            "active_process_count": process_count,
        }

    def get_device_info(self) -> dict[str, Any]:
        driver_version = self._decode_nvml_string(
            self._safe_nvml_call(pynvml.nvmlSystemGetDriverVersion)
        )

        cuda_driver_version = self._safe_nvml_call(
            pynvml.nvmlSystemGetCudaDriverVersion
        )

        compute_capability = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetCudaComputeCapability(self.handle)
        )

        graphics_clock_mhz = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                self.handle,
                pynvml.NVML_CLOCK_GRAPHICS,
            )
        )

        sm_clock_mhz = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                self.handle,
                pynvml.NVML_CLOCK_SM,
            )
        )

        memory_clock_mhz = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetClockInfo(
                self.handle,
                pynvml.NVML_CLOCK_MEM,
            )
        )

        performance_state = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetPerformanceState(self.handle)
        )

        fan_speed_percent = self._safe_nvml_call(
            lambda: pynvml.nvmlDeviceGetFanSpeed(self.handle)
        )

        power_limit_watts = self._safe_nvml_call(
            lambda: round(
                pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000,
                2,
            )
        )

        return {
            "gpu": self.get_gpu_name(),
            "driver_version": driver_version,
            "cuda_driver_version": cuda_driver_version,
            "pytorch_cuda_version": torch.version.cuda,
            "compute_capability": compute_capability,
            "graphics_clock_mhz": graphics_clock_mhz,
            "sm_clock_mhz": sm_clock_mhz,
            "memory_clock_mhz": memory_clock_mhz,
            "performance_state": performance_state,
            "fan_speed_percent": fan_speed_percent,
            "power_limit_watts": power_limit_watts,
        }

    def get_processes(self) -> dict[str, Any]:
        process_map: dict[int, dict[str, Any]] = {}

        process_sources = [
            {
                "type": "compute",
                "getter": pynvml.nvmlDeviceGetComputeRunningProcesses,
            },
            {
                "type": "graphics",
                "getter": pynvml.nvmlDeviceGetGraphicsRunningProcesses,
            },
        ]

        for source in process_sources:
            processes = self._safe_nvml_call(
                lambda source=source: source["getter"](self.handle),
                default=[],
            )

            for process in processes:
                pid = process.pid

                used_memory_mb = self._bytes_to_mb(
                    getattr(process, "usedGpuMemory", None)
                )

                process_details = self._read_process_details(pid)

                if pid not in process_map:
                    process_map[pid] = {
                        "pid": pid,
                        "name": process_details["name"],
                        "command": process_details["command"],
                        "types": [],
                        "used_memory_mb": used_memory_mb,
                    }

                process_map[pid]["types"].append(source["type"])

                if used_memory_mb is not None:
                    current_memory = process_map[pid]["used_memory_mb"] or 0
                    process_map[pid]["used_memory_mb"] = max(
                        current_memory,
                        used_memory_mb,
                    )

        processes_list = list(process_map.values())

        processes_list.sort(
            key=lambda item: item["used_memory_mb"]
            if item["used_memory_mb"] is not None
            else 0,
            reverse=True,
        )

        return {
            "count": len(processes_list),
            "processes": processes_list,
        }

    def get_gpu_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)

        return "No CUDA GPU detected"

    def _safe_nvml_call(self, callable_func, default=None):
        try:
            return callable_func()
        except pynvml.NVMLError:
            return default

    def _decode_nvml_string(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        return value

    def _bytes_to_mb(self, value):
        if value is None:
            return None

        try:
            if value == pynvml.NVML_VALUE_NOT_AVAILABLE:
                return None
        except AttributeError:
            pass

        return round(value / 1024**2, 2)

    def _read_process_details(self, pid: int) -> dict[str, str]:
        proc_path = Path(f"/proc/{pid}")

        process_name = "unknown"
        command = ""

        try:
            comm_path = proc_path / "comm"

            if comm_path.exists():
                process_name = comm_path.read_text().strip()

        except Exception:
            pass

        try:
            cmdline_path = proc_path / "cmdline"

            if cmdline_path.exists():
                raw_cmdline = cmdline_path.read_bytes()
                command = (
                    raw_cmdline.replace(b"\x00", b" ")
                    .decode("utf-8", errors="replace")
                    .strip()
                )

        except Exception:
            pass

        return {
            "name": process_name,
            "command": command,
        }
