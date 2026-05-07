NVCC ?= nvcc
TARGET := vector_add
SRC := vector_add.cu

NVCCFLAGS := -O2 -std=c++17

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET) 16777216 256 100

clean:
	rm -f $(TARGET)
