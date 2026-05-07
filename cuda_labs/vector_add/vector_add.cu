/*
    vector_add.cu

    This is a beginner-friendly CUDA C++ program.

    Goal:
    -----
    Add two arrays together on the GPU.

    Example:
        a = [1, 1, 1, 1]
        b = [2, 2, 2, 2]
        c = [3, 3, 3, 3]

    But instead of using a normal CPU loop, we ask the GPU to launch
    thousands or millions of tiny workers called "threads."

    Each GPU thread handles one array index.

    Conceptually:
        thread 0 computes c[0] = a[0] + b[0]
        thread 1 computes c[1] = a[1] + b[1]
        thread 2 computes c[2] = a[2] + b[2]
        ...
*/

#include <cuda_runtime.h>   // Gives us CUDA functions like cudaMalloc, cudaMemcpy, cudaFree

#include <algorithm>        // Gives us std::max
#include <cmath>            // Gives us std::fabs
#include <cstdlib>          // General C/C++ utilities
#include <iostream>         // Gives us std::cout and std::cerr
#include <string>           // Gives us std::stoi for converting command-line text to numbers
#include <vector>           // Gives us std::vector for CPU-side arrays


/*
    CUDA_CHECK macro
    ----------------

    CUDA functions return a value of type cudaError_t.

    If the CUDA function succeeds, it returns cudaSuccess.

    If something goes wrong, we want to print a clear error and stop
    the program instead of failing silently.

    Example:
        CUDA_CHECK(cudaMalloc(&device_a, bytes));

    Means:
        "Run cudaMalloc. If it fails, print the CUDA error and return 1."

    Note:
        This macro is written for use inside main(), because it returns 1
        when something fails.
*/
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(error) << std::endl;      \
            return 1;                                                          \
        }                                                                      \
    } while (0)


/*
    __global__
    ----------

    This keyword means:

        "This function runs on the GPU, but it is launched from the CPU."

    CUDA calls GPU functions "kernels."

    So this function is a CUDA kernel.

    Parameters:
    -----------
    const float* a
        Pointer to array A in GPU memory.

    const float* b
        Pointer to array B in GPU memory.

    float* c
        Pointer to output array C in GPU memory.

    int n
        Number of elements in each array.

    Important:
    ----------
    These pointers point to GPU memory, not normal CPU memory.
*/
__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    /*
        CUDA thread indexing
        --------------------

        The GPU launches threads in groups called "blocks."

        Each block has many threads.

        CUDA gives every thread a few built-in variables:

            blockIdx.x
                Which block am I in?

            blockDim.x
                How many threads are in each block?

            threadIdx.x
                Which thread am I inside this block?

        To calculate a globally unique index for each thread:

            index = blockIdx.x * blockDim.x + threadIdx.x

        Example:
            threads_per_block = 256

            block 0, thread 0   -> index 0
            block 0, thread 1   -> index 1
            ...
            block 0, thread 255 -> index 255

            block 1, thread 0   -> index 256
            block 1, thread 1   -> index 257
            ...
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    /*
        Boundary check
        --------------

        We may launch slightly more GPU threads than the number of elements.

        Example:
            n = 1000
            threads_per_block = 256

            blocks = ceil(1000 / 256) = 4

            4 blocks * 256 threads = 1024 threads

        But we only have 1000 array elements.

        So threads with index 1000 through 1023 must do nothing.

        That is why we check:

            if (index < n)
    */
    if (index < n) {
        /*
            This is the actual parallel work.

            Each GPU thread computes one element.

            This line runs many times in parallel across many GPU threads.
        */
        c[index] = a[index] + b[index];
    }
}


int main(int argc, char* argv[]) {
    /*
        Default configuration
        ---------------------

        n:
            Number of elements in the arrays.

            16 * 1024 * 1024 = 16,777,216 floats.

        threads_per_block:
            Number of GPU threads per CUDA block.

            256 is a very common beginner/default value.

        iterations:
            Number of times we run the kernel for timing.

            We run it many times because one single kernel launch may be
            too fast to measure cleanly.
    */
    int n = 16 * 1024 * 1024;
    int threads_per_block = 256;
    int iterations = 100;

    /*
        Command-line arguments
        ----------------------

        This lets us run the program with custom values.

        Example:
            ./vector_add 1000000 256 50

        Means:
            n = 1,000,000 elements
            threads_per_block = 256
            iterations = 50
    */
    if (argc >= 2) {
        n = std::stoi(argv[1]);
    }

    if (argc >= 3) {
        threads_per_block = std::stoi(argv[2]);
    }

    if (argc >= 4) {
        iterations = std::stoi(argv[3]);
    }

    /*
        Basic input validation
        ----------------------

        We do not want zero or negative values because that would make
        no sense for array sizes, thread counts, or iterations.
    */
    if (n <= 0 || threads_per_block <= 0 || iterations <= 0) {
        std::cerr << "Usage: ./vector_add [num_elements] [threads_per_block] [iterations]\n";
        return 1;
    }

    /*
        Calculate memory size
        ---------------------

        Each float uses 4 bytes.

        If n = 16,777,216:

            bytes = 16,777,216 * 4
                  = 67,108,864 bytes
                  ≈ 64 MB

        We need this many bytes for each array.
    */
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    /*
        CPU-side arrays
        ---------------

        These arrays live in normal system RAM.

        host_a:
            Input array A on CPU

        host_b:
            Input array B on CPU

        host_c:
            Output array C on CPU

        The word "host" usually means CPU side in CUDA programming.

        The word "device" usually means GPU side.
    */
    std::vector<float> host_a(n);
    std::vector<float> host_b(n);
    std::vector<float> host_c(n);

    /*
        Fill the CPU arrays
        -------------------

        We make every value predictable:

            host_a[i] = 1.0
            host_b[i] = 2.0

        So the correct answer should always be:

            host_c[i] = 3.0
    */
    for (int i = 0; i < n; i++) {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    /*
        GPU-side pointers
        -----------------

        These will point to memory allocated on the GPU.

        At this point they are nullptr because we have not allocated
        GPU memory yet.
    */
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    /*
        Allocate GPU memory
        -------------------

        cudaMalloc allocates memory in GPU VRAM.

        This is similar in spirit to malloc() in C, but the memory lives
        on the GPU.

        We allocate space for:
            device_a
            device_b
            device_c
    */
    CUDA_CHECK(cudaMalloc(&device_a, bytes));
    CUDA_CHECK(cudaMalloc(&device_b, bytes));
    CUDA_CHECK(cudaMalloc(&device_c, bytes));

    /*
        Copy input data from CPU RAM to GPU VRAM
        ----------------------------------------

        The GPU cannot directly use our std::vector CPU arrays.

        We have to copy the data to GPU memory first.

        cudaMemcpyHostToDevice means:
            CPU memory -> GPU memory
    */
    CUDA_CHECK(cudaMemcpy(
        device_a,
        host_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        device_b,
        host_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    /*
        Calculate how many CUDA blocks we need
        --------------------------------------

        We know:
            n = total number of array elements
            threads_per_block = number of threads in each block

        We want enough blocks so every element gets one thread.

        Formula:
            blocks = ceil(n / threads_per_block)

        In integer math, this common trick does ceiling division:

            (n + threads_per_block - 1) / threads_per_block

        Example:
            n = 1000
            threads_per_block = 256

            blocks = (1000 + 256 - 1) / 256
                   = 1255 / 256
                   = 4
    */
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    /*
        Warm-up kernel launch
        ---------------------

        The first CUDA operation can be slower because the CUDA runtime
        may initialize context/state.

        We run the kernel once before timing so our timing is cleaner.

        Syntax:
            kernel_name<<<blocks, threads_per_block>>>(arguments);

        The triple angle brackets are CUDA's kernel launch syntax.

        Meaning:
            Launch vector_add_kernel on the GPU using:
                blocks blocks
                threads_per_block threads in each block
    */
    vector_add_kernel<<<blocks, threads_per_block>>>(
        device_a,
        device_b,
        device_c,
        n
    );

    /*
        Check whether the kernel launch itself failed.

        Kernel launch errors are not always caught immediately unless
        we ask CUDA for the last error.
    */
    CUDA_CHECK(cudaGetLastError());

    /*
        Wait for the GPU to finish the warm-up kernel.

        CUDA operations are often asynchronous, which means the CPU can
        continue before the GPU is done.

        cudaDeviceSynchronize forces the CPU to wait.
    */
    CUDA_CHECK(cudaDeviceSynchronize());

    /*
        CUDA events
        -----------

        CUDA events are used to measure GPU time.

        This is better than normal CPU timers for GPU kernel timing.

        start:
            Event recorded before the timed workload.

        stop:
            Event recorded after the timed workload.
    */
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /*
        Record the start event on the GPU timeline.
    */
    CUDA_CHECK(cudaEventRecord(start));

    /*
        Run the kernel many times
        -------------------------

        We are timing only the GPU kernel execution here.

        We are not timing:
            - CPU array creation
            - cudaMalloc
            - CPU -> GPU copy
            - GPU -> CPU copy

        This isolates the GPU computation.
    */
    for (int i = 0; i < iterations; i++) {
        vector_add_kernel<<<blocks, threads_per_block>>>(
            device_a,
            device_b,
            device_c,
            n
        );
    }

    /*
        Check if any of the kernel launches failed.
    */
    CUDA_CHECK(cudaGetLastError());

    /*
        Record the stop event after the kernel launches.
    */
    CUDA_CHECK(cudaEventRecord(stop));

    /*
        Wait until the stop event is reached.

        This means all timed GPU work is finished.
    */
    CUDA_CHECK(cudaEventSynchronize(stop));

    /*
        Calculate elapsed GPU time between start and stop.

        The result is in milliseconds.
    */
    float total_kernel_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_kernel_ms,
        start,
        stop
    ));

    /*
        Copy result back from GPU VRAM to CPU RAM
        -----------------------------------------

        device_c has the result, but it lives on the GPU.

        We copy it back into host_c so the CPU can verify correctness.

        cudaMemcpyDeviceToHost means:
            GPU memory -> CPU memory
    */
    CUDA_CHECK(cudaMemcpy(
        host_c.data(),
        device_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    /*
        Verify result
        -------------

        Since:
            a[i] = 1.0
            b[i] = 2.0

        We expect:
            c[i] = 3.0

        Floating-point math can sometimes have tiny rounding differences,
        so we calculate max_error instead of requiring exact equality.
    */
    float max_error = 0.0f;

    for (int i = 0; i < n; i++) {
        float expected = 3.0f;
        float error = std::fabs(host_c[i] - expected);

        max_error = std::max(max_error, error);
    }

    /*
        If max_error is extremely small, the test passed.
    */
    bool passed = max_error < 0.00001f;

    /*
        Free GPU memory
        ---------------

        Anything allocated with cudaMalloc should be released with cudaFree.

        This is like cleaning up after yourself so the GPU memory is not leaked.
    */
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));

    /*
        Destroy CUDA events
        -------------------

        We created these with cudaEventCreate, so we clean them up.
    */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    /*
        Print JSON-style output
        -----------------------

        This makes the program easier to call later from Python/FastAPI.

        The dashboard can eventually execute this binary and parse the output.
    */
    std::cout << "{\n";
    std::cout << "  \"lab\": \"vector_add\",\n";
    std::cout << "  \"elements\": " << n << ",\n";
    std::cout << "  \"threads_per_block\": " << threads_per_block << ",\n";
    std::cout << "  \"blocks\": " << blocks << ",\n";
    std::cout << "  \"iterations\": " << iterations << ",\n";
    std::cout << "  \"total_kernel_ms\": " << total_kernel_ms << ",\n";
    std::cout << "  \"avg_kernel_ms\": " << total_kernel_ms / iterations << ",\n";
    std::cout << "  \"max_error\": " << max_error << ",\n";
    std::cout << "  \"passed\": " << (passed ? "true" : "false") << "\n";
    std::cout << "}\n";

    /*
        Exit code
        ---------

        return 0 means success.

        return 1 means failure.

        This matters later if Python or CI/CD runs this program.
    */
    return passed ? 0 : 1;
}
