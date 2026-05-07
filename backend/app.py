from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.services.benchmark_runner import BenchmarkRunner
from backend.services.gpu_monitor import GpuMonitor
from backend.services.cuda_lab_runner import CudaLabRunner


app = FastAPI(title="CUDA Ops Dashboard")

app.mount("/static", StaticFiles(directory="backend/static"), name="static")
templates = Jinja2Templates(directory="backend/templates")


gpu_monitor = GpuMonitor(gpu_index=0)
benchmark_runner = BenchmarkRunner()
cuda_lab_runner = CudaLabRunner()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/gpu/metrics")
def gpu_metrics():
    metrics = gpu_monitor.get_metrics()
    metrics["benchmark"] = benchmark_runner.get_status()
    return metrics


@app.get("/gpu/device-info")
def gpu_device_info():
    return gpu_monitor.get_device_info()


@app.get("/gpu/processes")
def gpu_processes():
    return gpu_monitor.get_processes()


@app.get("/gpu/benchmark/status")
def benchmark_status():
    return benchmark_runner.get_status()

@app.get("/cuda-labs/vector-add/info")
def vector_add_info():
    return cuda_lab_runner.get_vector_add_info()

@app.post("/gpu/benchmark/start")
def start_benchmark(
    size: int = Query(
        default=8000,
        ge=BenchmarkRunner.MIN_MATRIX_SIZE,
        le=BenchmarkRunner.MAX_MATRIX_SIZE,
    ),
    iterations: int = Query(
        default=25,
        ge=BenchmarkRunner.MIN_ITERATIONS,
        le=BenchmarkRunner.MAX_ITERATIONS,
    ),
):
    return benchmark_runner.start(size=size, iterations=iterations)


@app.post("/gpu/benchmark/stop")
def stop_benchmark():
    return benchmark_runner.stop()

@app.post("/cuda-labs/vector-add/run")
def run_vector_add_lab(
    elements: int = Query(
        default=16_777_216,
        ge=CudaLabRunner.MIN_ELEMENTS,
        le=CudaLabRunner.MAX_ELEMENTS,
    ),
    threads_per_block: int = Query(
        default=256,
        ge=CudaLabRunner.MIN_THREADS_PER_BLOCK,
        le=CudaLabRunner.MAX_THREADS_PER_BLOCK,
    ),
    iterations: int = Query(
        default=100,
        ge=CudaLabRunner.MIN_ITERATIONS,
        le=CudaLabRunner.MAX_ITERATIONS,
    ),
):
    return cuda_lab_runner.run_vector_add(
        elements=elements,
        threads_per_block=threads_per_block,
        iterations=iterations,
    )
