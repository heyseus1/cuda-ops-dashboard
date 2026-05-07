NVCC ?= nvcc
TARGET := mandelbrot
SRC := mandelbrot.cu

NVCCFLAGS := -O2 -std=c++17

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET) output.bmp 960 540 500 -0.5 0.0 1.0

clean:
	rm -f $(TARGET) output.bmp
