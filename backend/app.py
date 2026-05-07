from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pynvml
import torch
import time


app = FastAPI(title="CUDA Ops Dashboard")

app.mount("/static", StaticFiles(directory="backend/static"), name="static")
templates = Jinja2Templates(directory="backend/templates")

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


benchmark_state = {
    "running": False,
    "last_seconds": None,
    "last_size": None,
    "last_error": None,
}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/gpu/metrics")
def gpu_metrics():
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(
        handle,
        pynvml.NVML_TEMPERATURE_GPU,
    )
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

    memory_used_mb = round(memory.used / 1024**2, 2)
    memory_total_mb = round(memory.total / 1024**2, 2)
    memory_percent = round((memory.used / memory.total) * 100, 2)

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU detected",
        "cuda_available": torch.cuda.is_available(),
        "utilization_percent": utilization.gpu,
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
        "memory_percent": memory_percent,
        "temperature_c": temperature,
        "power_watts": round(power, 2),
        "benchmark": benchmark_state,
    }


def run_benchmark(size: int):
    benchmark_state["running"] = True
    benchmark_state["last_error"] = None
    benchmark_state["last_size"] = size

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")

        x = torch.randn(size, size, device="cuda")
        y = torch.randn(size, size, device="cuda")

        start = time.time()

        _ = x @ y

        torch.cuda.synchronize()

        benchmark_state["last_seconds"] = round(time.time() - start, 4)

    except Exception as exc:
        benchmark_state["last_error"] = str(exc)

    finally:
        benchmark_state["running"] = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/gpu/benchmark")
def start_benchmark(background_tasks: BackgroundTasks, size: int = 10000):
    if benchmark_state["running"]:
        return {
            "status": "already_running",
            "benchmark": benchmark_state,
        }

    background_tasks.add_task(run_benchmark, size)

    return {
        "status": "started",
        "size": size,
    }
