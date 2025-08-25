USE_CUDA ?= OFF
USE_SYCL ?= OFF

$(info USE_CUDA=$(USE_CUDA))
$(info USE_SYCL=$(USE_SYCL))

CXX?=gcc

ifeq ($(USE_SYCL), ON)
ICPX?=icpx
endif

# source code
ALL_C_CPP_SOURCES=$(wildcard src/*.c src/*.cc src/*.cpp)
ALL_C_CPP_SOURCES+=$(wildcard vendor/*.c vendor/*.cc vendor/*.cpp)

ALL_CUDA_SOURCES:=
ifeq ($(USE_CUDA), ON)
ALL_CUDA_SOURCES+=$(wildcard src/*.cu)
ALL_CUDA_SOURCES+=$(wildcard vendor/*.cu)
endif

# --- Source Code Definition ---
# 1. Gather all source files from all relevant directories.
# ALL_SOURCES := $(wildcard src/*.c src/*.cc src/*.cpp src/*.cu)
# ALL_SOURCES += $(wildcard vendor/*.c vendor/*.cc vendor/*.cpp vendor/*.cu)
ALL_SOURCES := $(ALL_C_CPP_SOURCES)
ifeq ($(USE_CUDA), ON)
	ALL_SOURCES += $(ALL_CUDA_SOURCES)
endif

# 2. Define files that are entry points for other targets (e.g., tests).
MAIN_FILE=main.cpp
TEST_FILE=test.cpp

# 3. The final list of sources for the main binary is everything except the test files.
SOURCES ?= $(ALL_SOURCES)
SOURCES += $(MAIN_FILE)

# build directory
BUILD_DIR=build
OBJECTS=$(SOURCES:%=$(BUILD_DIR)/%.o)

BINARY_MAIN=$(BUILD_DIR)/main
# @todo
BINARY_TEST=$(BUILD_DIR)/test

# CFLAGS=-g -Wall -Wpointer-arith -Werror -O3 -ffast-math -Ivendor -std=c++2a -DUSE_CUDA=$(USE_CUDA)
CFLAGS=-g -Wall -Wpointer-arith -O3 -ffast-math -Ivendor -std=c++2a
CFLAGS+=-fopenmp -mf16c -mavx2 -mfma

LDFLAGS=-lm
LDFLAGS+=-fopenmp

ifeq ($(USE_CUDA), ON)
	NVCC?=nvcc
  	CFLAGS+=-I/usr/local/cuda/include
	CFLAGS+=-DUSE_CUDA=$(USE_CUDA)
	LDFLAGS+=-lcudart
  	LDFLAGS+=-L/usr/local/cuda/lib64

	CUFLAGS+=-O2 -lineinfo -Ivendor
	CUFLAGS+=-DUSE_CUDA=$(USE_CUDA)
	CUFLAGS+=-rdc=true
	CUFLAGS+=-allow-unsupported-compiler # for recent CUDA versions
  CUFLAGS+=-gencode arch=compute_80,code=sm_80 --threads 2
endif

ifeq ($(USE_SYCL), ON)
	$(info "No Supported!")
endif

# 
all: $(BINARY_MAIN)

main: $(BINARY_MAIN)

# @todo
test: $(BINARY_TEST)

$(BINARY_MAIN): $(OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD_DIR)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

clean:
	rm -rf $(BUILD_DIR)
