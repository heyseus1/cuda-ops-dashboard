#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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
    This kernel renders one Mandelbrot pixel per CUDA thread.

    Thread mapping:
    - one CUDA thread handles one pixel
    - x = horizontal pixel coordinate
    - y = vertical pixel coordinate
*/
__global__ void mandelbrot_kernel(
    unsigned char* image,
    int width,
    int height,
    int max_iterations,
    double center_x,
    double center_y,
    double zoom
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    /*
        Map the pixel coordinate into the complex plane.

        A larger zoom means we are looking closer into the fractal.
    */
    double scaled_x = (x - (width / 2.0)) / (0.5 * zoom * width);
    double scaled_y = (y - (height / 2.0)) / (0.5 * zoom * height);

    /*
        Stretch the horizontal axis a bit so the full set looks better.
    */
    double c_re = scaled_x * 3.5 + center_x;
    double c_im = scaled_y * 2.0 + center_y;

    double z_re = 0.0;
    double z_im = 0.0;

    int iteration = 0;

    while ((z_re * z_re + z_im * z_im) <= 4.0 && iteration < max_iterations) {
        double new_re = z_re * z_re - z_im * z_im + c_re;
        double new_im = 2.0 * z_re * z_im + c_im;

        z_re = new_re;
        z_im = new_im;
        iteration++;
    }

    int pixel_index = (y * width + x) * 3;

    /*
        Color the pixel.

        If the point never escaped, we color it black.
        Otherwise, we use a simple smooth palette.
    */
    if (iteration == max_iterations) {
        image[pixel_index + 0] = 0;
        image[pixel_index + 1] = 0;
        image[pixel_index + 2] = 0;
        return;
    }

    double t = static_cast<double>(iteration) / static_cast<double>(max_iterations);

    unsigned char red = static_cast<unsigned char>(
        9.0 * (1.0 - t) * t * t * t * 255.0
    );

    unsigned char green = static_cast<unsigned char>(
        15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0
    );

    unsigned char blue = static_cast<unsigned char>(
        8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0
    );

    image[pixel_index + 0] = red;
    image[pixel_index + 1] = green;
    image[pixel_index + 2] = blue;
}

/*
    Write a very simple 24-bit BMP image.

    BMP is useful here because:
    - it is easy to write manually
    - browsers can display it
    - we do not need external image libraries
*/
bool write_bmp(
    const std::string& output_path,
    const std::vector<unsigned char>& rgb_data,
    int width,
    int height
) {
    std::ofstream file(output_path, std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    int row_stride = width * 3;
    int row_padding = (4 - (row_stride % 4)) % 4;
    int pixel_data_size = (row_stride + row_padding) * height;
    int file_size = 14 + 40 + pixel_data_size;

    /*
        BMP file header
    */
    file.put('B');
    file.put('M');

    auto write_u16 = [&file](std::uint16_t value) {
        file.put(static_cast<char>(value & 0xFF));
        file.put(static_cast<char>((value >> 8) & 0xFF));
    };

    auto write_u32 = [&file](std::uint32_t value) {
        file.put(static_cast<char>(value & 0xFF));
        file.put(static_cast<char>((value >> 8) & 0xFF));
        file.put(static_cast<char>((value >> 16) & 0xFF));
        file.put(static_cast<char>((value >> 24) & 0xFF));
    };

    write_u32(file_size);
    write_u16(0);
    write_u16(0);
    write_u32(54);

    /*
        BMP info header
    */
    write_u32(40);
    write_u32(static_cast<std::uint32_t>(width));
    write_u32(static_cast<std::uint32_t>(height));
    write_u16(1);
    write_u16(24);
    write_u32(0);
    write_u32(static_cast<std::uint32_t>(pixel_data_size));
    write_u32(2835);
    write_u32(2835);
    write_u32(0);
    write_u32(0);

    /*
        BMP stores rows bottom-up and pixels in BGR order.
    */
    std::vector<unsigned char> padding(row_padding, 0);

    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * 3;

            unsigned char r = rgb_data[index + 0];
            unsigned char g = rgb_data[index + 1];
            unsigned char b = rgb_data[index + 2];

            file.put(static_cast<char>(b));
            file.put(static_cast<char>(g));
            file.put(static_cast<char>(r));
        }

        if (row_padding > 0) {
            file.write(reinterpret_cast<const char*>(padding.data()), row_padding);
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    /*
        Usage:
        ./mandelbrot <output_path> <width> <height> <max_iterations> <center_x> <center_y> <zoom>
    */
    if (argc < 8) {
        std::cerr << "Usage: ./mandelbrot <output_path> <width> <height> <max_iterations> <center_x> <center_y> <zoom>\n";
        return 1;
    }

    std::string output_path = argv[1];
    int width = std::stoi(argv[2]);
    int height = std::stoi(argv[3]);
    int max_iterations = std::stoi(argv[4]);
    double center_x = std::stod(argv[5]);
    double center_y = std::stod(argv[6]);
    double zoom = std::stod(argv[7]);

    if (width <= 0 || height <= 0 || max_iterations <= 0 || zoom <= 0.0) {
        std::cerr << "Invalid input values.\n";
        return 1;
    }

    size_t image_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;

    std::vector<unsigned char> host_image(image_bytes, 0);
    unsigned char* device_image = nullptr;

    CUDA_CHECK(cudaMalloc(&device_image, image_bytes));

    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    mandelbrot_kernel<<<blocks, threads>>>(
        device_image,
        width,
        height,
        max_iterations,
        center_x,
        center_y,
        zoom
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(
        host_image.data(),
        device_image,
        image_bytes,
        cudaMemcpyDeviceToHost
    ));

    bool wrote_file = write_bmp(output_path, host_image, width, height);

    CUDA_CHECK(cudaFree(device_image));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (!wrote_file) {
        std::cerr << "Failed to write BMP output file.\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n";
    std::cout << "  \"lab\": \"mandelbrot\",\n";
    std::cout << "  \"width\": " << width << ",\n";
    std::cout << "  \"height\": " << height << ",\n";
    std::cout << "  \"pixels\": " << (width * height) << ",\n";
    std::cout << "  \"max_iterations\": " << max_iterations << ",\n";
    std::cout << "  \"center_x\": " << center_x << ",\n";
    std::cout << "  \"center_y\": " << center_y << ",\n";
    std::cout << "  \"zoom\": " << zoom << ",\n";
    std::cout << "  \"threads_x\": " << threads.x << ",\n";
    std::cout << "  \"threads_y\": " << threads.y << ",\n";
    std::cout << "  \"blocks_x\": " << blocks.x << ",\n";
    std::cout << "  \"blocks_y\": " << blocks.y << ",\n";
    std::cout << "  \"kernel_ms\": " << total_kernel_ms << ",\n";
    std::cout << "  \"output_path\": \"" << output_path << "\"\n";
    std::cout << "}\n";

    return 0;
}
