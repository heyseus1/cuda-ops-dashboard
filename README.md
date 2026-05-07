# CUDA Ops Dashboard

CUDA Ops Dashboard is a local NVIDIA GPU observability and CUDA development lab.

It provides a browser-based dashboard for:

- Monitoring live NVIDIA GPU telemetry
- Viewing GPU utilization, VRAM, temperature, power, clocks, and process usage
- Running controlled CUDA workloads
- Starting and stopping benchmark workloads
- Executing custom CUDA C++ kernels from the UI
- Viewing raw JSON output and human-readable explanations of CUDA lab results

This project was built and tested on:

```text
Windows + WSL Ubuntu
NVIDIA GeForce RTX 3090
Python 3.12
FastAPI
PyTorch
NVIDIA Management Library
CUDA C++
```

---

## Demo

![CUDA Ops Dashboard Demo](assets/cuda.gif)

---

## Project Structure

```text
cuda-ops-dashboard/
├── assets/
│   └── cuda-dashboard-demo.gif
├── backend/
│   ├── app.py
│   ├── services/
│   │   ├── benchmark_runner.py
│   │   ├── cuda_lab_runner.py
│   │   └── gpu_monitor.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── cuda_labs/
│   ├── README.md
│   └── vector_add/
│       ├── Makefile
│       └── vector_add.cu
├── scripts/
│   └── gpu_check.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Application Architecture

```text
Browser UI
   |
   v
FastAPI Backend
   |
   +--> GPU Metrics
   |       |
   |       +--> NVIDIA Management Library
   |       +--> PyTorch CUDA status
   |
   +--> Benchmark Runner
   |       |
   |       +--> PyTorch CUDA matrix multiplication workload
   |       +--> start / stop controls
   |       +--> benchmark progress state
   |
   +--> GPU Process Monitor
   |       |
   |       +--> active compute / graphics GPU processes
   |       +--> process memory usage
   |
   +--> CUDA Lab Runner
           |
           +--> compiled CUDA C++ binaries
           +--> vector_add CUDA kernel
           +--> JSON output parsing
```

---

## Dashboard Tabs

### Dashboard

The main dashboard displays:

- GPU name
- CUDA availability
- GPU utilization
- VRAM used, free, and total
- Temperature
- Power draw
- Power limit
- Power headroom
- Graphics clock
- Memory clock
- Active GPU process count
- Live charts
- Benchmark controls
- Benchmark progress
- Benchmark logs

---

### Process Monitor

The process monitor displays:

- NVIDIA driver version
- CUDA driver version
- PyTorch CUDA version
- Compute capability
- Performance state
- Clock speeds
- Fan speed if exposed by NVML
- Power limit
- Active GPU consumers

On WSL, some Windows-side GPU processes may appear as `unknown` because they may not map cleanly into Linux `/proc`.

---

### CUDA Kernel Lab

The CUDA Kernel Lab currently includes:

```text
Vector Add Kernel
```

This runs a compiled CUDA C++ executable from the web UI.

The lab displays:

- Run status
- Passed / failed result
- Average GPU kernel time
- Wall time
- Element count
- CUDA block count
- Threads per block
- Max error
- Raw JSON output
- Human-readable result summary

---

# Setup Path 1: Windows + WSL Ubuntu

This is the main setup path for this project.

---

## 1. Enable CPU Virtualization in BIOS

WSL 2 requires CPU virtualization.

Restart your PC and enter BIOS / UEFI.

Common BIOS keys:

```text
DEL
F2
F10
F12
ESC
```

Look for one of these settings:

```text
Intel Virtualization Technology
Intel VT-x
AMD-V
SVM Mode
Virtualization Technology
```

Enable it.

Save BIOS settings and reboot into Windows.

---

## 2. Enable WSL and Virtual Machine Platform

Open **PowerShell as Administrator**.

Run:

```powershell
wsl --install --no-distribution
```

If that does not work, enable the Windows features manually:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Then reboot Windows.

After reboot:

```powershell
wsl --set-default-version 2
```

---

## 3. Install Ubuntu

Install Ubuntu:

```powershell
wsl --install -d Ubuntu
```

Or install Ubuntu 22.04 specifically:

```powershell
wsl --install -d Ubuntu-22.04
```

Launch Ubuntu and create your Linux username and password.

---

## 4. Verify WSL Version

From PowerShell:

```powershell
wsl --list --verbose
```

You want Ubuntu to show:

```text
VERSION 2
```

If it shows version 1:

```powershell
wsl --set-version Ubuntu 2
```

---

## 5. Install NVIDIA Windows Driver

Install the latest NVIDIA Windows driver for your GPU.

Reboot Windows after installation.

Then open Ubuntu / WSL and run:

```bash
nvidia-smi
```

Expected result:

```text
NVIDIA GeForce RTX 3090
```

If `nvidia-smi` works inside WSL, Ubuntu can see your GPU.

---

# Setup Path 2: Native Linux

This project can also run on native Ubuntu Linux with an NVIDIA GPU.

You need:

- NVIDIA driver installed
- `nvidia-smi` working
- Python 3.12
- Python virtual environment support
- CUDA Toolkit if you want to compile CUDA C++ labs

Verify GPU visibility:

```bash
nvidia-smi
```

---

# Linux Setup Steps

Run these steps inside Ubuntu, whether you are using WSL or native Linux.

---

## 1. Update Ubuntu

```bash
sudo apt update
sudo apt upgrade -y
```

---

## 2. Install Base Development Tools

```bash
sudo apt install -y \
  build-essential \
  git \
  make \
  curl \
  wget \
  software-properties-common
```

---

## 3. Install Python 3.12

Check if Python 3.12 exists:

```bash
python3.12 --version
```

If Python 3.12 is missing, install it.

On Ubuntu versions that do not include Python 3.12 by default:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

Verify:

```bash
python3.12 --version
```

Expected:

```text
Python 3.12.x
```

---

## 4. Clone the Repository

Using SSH:

```bash
cd ~
git clone git@github.com:heyseus1/cuda-ops-dashboard.git
cd cuda-ops-dashboard
```

Using HTTPS:

```bash
cd ~
git clone https://github.com/heyseus1/cuda-ops-dashboard.git
cd cuda-ops-dashboard
```

---

## 5. Create a Python Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Verify PyTorch CUDA Access

Run:

```bash
python scripts/gpu_check.py
```

Expected output:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

If CUDA is unavailable, check:

```bash
nvidia-smi
```

If `nvidia-smi` fails, fix the NVIDIA driver / WSL GPU setup before continuing.

---

# CUDA Toolkit Setup

PyTorch can use CUDA without the full CUDA Toolkit installed.

However, this project includes custom CUDA C++ labs. To compile `.cu` files, you need `nvcc`.

Check for `nvcc`:

```bash
nvcc --version
```

If `nvcc` is missing, install the CUDA Toolkit.

Basic Ubuntu option:

```bash
sudo apt install -y nvidia-cuda-toolkit
```

Verify:

```bash
nvcc --version
```

---

# Build CUDA Labs

The CUDA source code is committed to Git.

Compiled CUDA binaries are intentionally ignored and must be built locally.

Build the Vector Add lab:

```bash
cd ~/cuda-ops-dashboard/cuda_labs/vector_add
make
```

Run it manually:

```bash
make run
```

Expected output:

```json
{
  "lab": "vector_add",
  "elements": 16777216,
  "threads_per_block": 256,
  "blocks": 65536,
  "iterations": 100,
  "total_kernel_ms": 28.0389,
  "avg_kernel_ms": 0.280389,
  "max_error": 0,
  "passed": true
}
```

Return to the repo root:

```bash
cd ~/cuda-ops-dashboard
```

---

# Run the Dashboard

Activate the virtual environment:

```bash
cd ~/cuda-ops-dashboard
source .venv/bin/activate
```

Start FastAPI:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 9000 --reload
```

Open the dashboard:

```text
http://localhost:9000
```

If using WSL, open that URL from your Windows browser.

---

# API Endpoints

## Dashboard

```text
GET /
```

Loads the local browser dashboard.

---

## GPU Metrics

```text
GET /gpu/metrics
```

Returns live GPU telemetry.

Example fields:

```json
{
  "gpu": "NVIDIA GeForce RTX 3090",
  "cuda_available": true,
  "utilization_percent": 10,
  "memory_used_mb": 1200.5,
  "memory_total_mb": 24576.0,
  "memory_free_mb": 23375.5,
  "memory_percent": 4.88,
  "temperature_c": 54,
  "power_watts": 98.25,
  "power_limit_watts": 390.0,
  "power_percent": 25.19,
  "power_headroom_watts": 291.75,
  "graphics_clock_mhz": 1800,
  "memory_clock_mhz": 9501,
  "active_process_count": 1
}
```

---

## GPU Device Info

```text
GET /gpu/device-info
```

Returns GPU device metadata.

---

## GPU Processes

```text
GET /gpu/processes
```

Returns active GPU compute and graphics processes.

---

## Benchmark Status

```text
GET /gpu/benchmark/status
```

Returns benchmark status and logs.

---

## Start Benchmark

```text
POST /gpu/benchmark/start?size=8000&iterations=25
```

Starts the PyTorch CUDA benchmark.

---

## Stop Benchmark

```text
POST /gpu/benchmark/stop
```

Requests a cooperative stop.

The benchmark stops safely between CUDA iterations.

---

## CUDA Vector Add Info

```text
GET /cuda-labs/vector-add/info
```

Returns CUDA lab configuration and binary status.

---

## Run CUDA Vector Add Lab

```text
POST /cuda-labs/vector-add/run?elements=16777216&threads_per_block=256&iterations=100
```

Runs the compiled CUDA C++ vector add binary and parses its JSON output.

---

# How the Vector Add CUDA Kernel Works

The vector add lab adds two arrays on the GPU.

Input:

```text
a[i] = 1.0
b[i] = 2.0
```

Output:

```text
c[i] = 3.0
```

CUDA kernel:

```cpp
__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        c[index] = a[index] + b[index];
    }
}
```

Each CUDA thread computes one output element.

Example:

```text
65,536 blocks * 256 threads per block = 16,777,216 logical CUDA threads
```

---

# Development Commands

Run the app:

```bash
cd ~/cuda-ops-dashboard
source .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 9000 --reload
```

Build CUDA labs:

```bash
cd ~/cuda-ops-dashboard/cuda_labs/vector_add
make
```

Run Vector Add manually:

```bash
cd ~/cuda-ops-dashboard/cuda_labs/vector_add
./vector_add 16777216 256 100
```

Clean CUDA binary:

```bash
cd ~/cuda-ops-dashboard/cuda_labs/vector_add
make clean
```

Update Python dependencies file:

```bash
cd ~/cuda-ops-dashboard
source .venv/bin/activate
pip freeze > requirements.txt
```

Check Git status:

```bash
git status
```

Check ignored files:

```bash
git status --ignored
```

---

# Git Tracking Notes

Commit these source files:

```text
cuda_labs/vector_add/vector_add.cu
cuda_labs/vector_add/Makefile
```

Do not commit compiled binaries:

```text
cuda_labs/vector_add/vector_add
*.o
```

The compiled binary is ignored because it is machine-specific and should be rebuilt locally.

---

# Troubleshooting

## WSL Error: HCS_E_HYPERV_NOT_INSTALLED

This usually means WSL 2 cannot create its virtual machine.

Check that:

- CPU virtualization is enabled in BIOS
- Virtual Machine Platform is enabled
- Windows Subsystem for Linux is enabled

PowerShell as Administrator:

```powershell
bcdedit /set hypervisorlaunchtype auto
```

Reboot Windows.

---

## WSL Error: Virtual Machine Platform Must Be Enabled

PowerShell as Administrator:

```powershell
wsl --install --no-distribution
```

Or manually:

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

Reboot Windows.

---

## Python Error: externally-managed-environment

Do not install Python packages globally.

Use a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## PyTorch Install Fails

Check Python version:

```bash
python --version
```

Recommended:

```text
Python 3.12
```

Avoid bleeding-edge Python versions for this project because PyTorch and GPU package support may lag behind new Python releases.

---

## CUDA Is Not Available in PyTorch

Check GPU visibility:

```bash
nvidia-smi
```

Then run:

```bash
python scripts/gpu_check.py
```

If `nvidia-smi` fails, fix the NVIDIA driver or WSL GPU setup first.

If `nvidia-smi` works but PyTorch CUDA fails, reinstall PyTorch for your environment.

---

## nvcc Not Found

PyTorch CUDA can work even if `nvcc` is missing.

Custom CUDA C++ labs require `nvcc`.

Install CUDA Toolkit:

```bash
sudo apt install -y nvidia-cuda-toolkit
```

Verify:

```bash
nvcc --version
```

---

## CUDA Kernel Lab Says Build Required

The UI runs this compiled binary:

```text
cuda_labs/vector_add/vector_add
```

If it is missing:

```bash
cd ~/cuda-ops-dashboard/cuda_labs/vector_add
make
```

Then refresh the dashboard.

---

## Active GPU Consumer Shows Unknown

This can happen under WSL.

The dashboard may see a GPU process through NVML, but if the process belongs to Windows, it may not map cleanly to Linux `/proc`.

This is expected for some Windows-side GPU consumers.

---

# Cleanup

Stop the FastAPI server:

```text
CTRL+C
```

Deactivate the Python virtual environment:

```bash
deactivate
```

Clean CUDA lab binary:

```bash
cd ~/cuda-ops-dashboard/cuda_labs/vector_add
make clean
```

---

# Roadmap

Planned future improvements:

- Add root-level Makefile
- Add CUDA image grayscale lab
- Add CUDA image blur lab
- Add CPU vs CUDA comparison lab
- Add Prometheus metrics endpoint
- Add Docker support with NVIDIA Container Toolkit
- Add Kubernetes deployment example
- Move inline JavaScript into `backend/static/app.js`
