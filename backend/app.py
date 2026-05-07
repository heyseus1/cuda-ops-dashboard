from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.services.benchmark_runner import BenchmarkRunner
from backend.services.cuda_lab_runner import CudaLabRunner
from backend.services.gpu_monitor import GpuMonitor
from backend.services.visual_lab_runner import VisualLabRunner


app = FastAPI(title="CUDA Ops Dashboard")

base_dir = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")
templates = Jinja2Templates(directory=str(base_dir / "templates"))

gpu_monitor = GpuMonitor()
benchmark_runner = BenchmarkRunner()
cuda_lab_runner = CudaLabRunner()
visual_lab_runner = VisualLabRunner()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
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
def gpu_benchmark_status():
    return benchmark_runner.get_status()


@app.post("/gpu/benchmark/start")
def gpu_benchmark_start(
    size: int = Query(default=8000, ge=512, le=16000),
    iterations: int = Query(default=25, ge=1, le=250),
):
    return benchmark_runner.start(size=size, iterations=iterations)


@app.post("/gpu/benchmark/stop")
def gpu_benchmark_stop():
    return benchmark_runner.stop()


@app.get("/cuda-labs/vector-add/info")
def vector_add_info():
    return cuda_lab_runner.get_vector_add_info()


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


@app.get("/cuda-visual/status")
def cuda_visual_status():
    return visual_lab_runner.status()


@app.post("/cuda-visual/start")
def cuda_visual_start(
    width: int = Query(
        default=960,
        ge=VisualLabRunner.MIN_WIDTH,
        le=VisualLabRunner.MAX_WIDTH,
    ),
    height: int = Query(
        default=540,
        ge=VisualLabRunner.MIN_HEIGHT,
        le=VisualLabRunner.MAX_HEIGHT,
    ),
    max_iterations: int = Query(
        default=500,
        ge=VisualLabRunner.MIN_ITERATIONS,
        le=VisualLabRunner.MAX_ITERATIONS,
    ),
    zoom_multiplier: float = Query(
        default=1.08,
        ge=VisualLabRunner.MIN_ZOOM_MULTIPLIER,
        le=VisualLabRunner.MAX_ZOOM_MULTIPLIER,
    ),
    interval_ms: int = Query(
        default=750,
        ge=VisualLabRunner.MIN_INTERVAL_MS,
        le=VisualLabRunner.MAX_INTERVAL_MS,
    ),
):
    return visual_lab_runner.start(
        width=width,
        height=height,
        max_iterations=max_iterations,
        zoom_multiplier=zoom_multiplier,
        interval_ms=interval_ms,
    )


@app.post("/cuda-visual/stop")
def cuda_visual_stop():
    return visual_lab_runner.stop()


@app.get("/cuda-visual/latest-image")
def cuda_visual_latest_image():
    if not visual_lab_runner.latest_image_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "not_found",
                "message": "No CUDA visual frame has been rendered yet.",
            },
        )

    return FileResponse(
        path=visual_lab_runner.latest_image_path,
        media_type="image/bmp",
        filename="live-mandelbrot.bmp",
    )
