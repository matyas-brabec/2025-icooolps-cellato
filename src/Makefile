CXX = g++
CXX_STD = c++20
NVCC = nvcc
CXXFLAGS = -std=$(CXX_STD) -Wall -Wextra -Wpedantic -O3 -I../include
NVCC_CC_BIN = $(shell which $(CXX))
NVCCFLAGS = -std=$(CXX_STD) -I../include -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -ccbin $(NVCC_CC_BIN) -Xcompiler -std=$(CXX_STD) -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -O3
CMAKE = cmake

CUDA_VERSION = 12.8

CUDA_HOME = /usr/local/cuda

INCLUDE_DIRS = -I../include -I$(CUDA_HOME)/include
LIB_DIRS = -L$(CUDA_HOME)/lib64
LIBS = -lcudart

KOKKOS_HOME_INCLUDE = $(DEP_DIR)/kokkos-install/include
KOKKOS_HOME_LIB = $(DEP_DIR)/kokkos-install/lib ${DEP_DIR}/kokkos-install/lib64

HALIDE_HOME_INCLUDE = $(DEP_DIR)/halide-install/include
HALIDE_HOME_LIB = $(DEP_DIR)/halide-install/lib ${DEP_DIR}/halide-install/lib64
HALIDE_LIBS = -lHalide

GRIDTOOLS_HOME_INCLUDE = $(DEP_DIR)/gridtools/include
GRIDTOOLS_HOME_LIB = $(DEP_DIR)/gridtools/lib ${DEP_DIR}/gridtools/lib64

CUDA_DRIVER_LIB = $(shell pkg-config --libs cuda-$(CUDA_VERSION))

ENABLE_KOKKOS = OFF
ENABLE_HALIDE = OFF
ENABLE_GRIDTOOLS = OFF

# Directories
SRC_DIR = .
OBJ_DIR = ../bin
BIN_DIR = ../bin
DEP_DIR = ../_deps
VENDOR_DIR = ../vendor
CUDA_DIRS = ../include/evaluators ../include/traversers/cuda ./game_of_life ./greenberg ./wire ./fire
TEST_DIR = $(SRC_DIR)

# Source and object files
MAIN_SRC = $(SRC_DIR)/main.cpp
MAIN_OBJ = $(OBJ_DIR)/main.o
TEST_SRC = $(SRC_DIR)/test_runner.cpp
TEST_OBJ = $(OBJ_DIR)/test_runner.o

# CUDA source and object files
CUDA_SRCS = $(foreach dir,$(CUDA_DIRS),$(wildcard $(dir)/*.cu))
CUDA_OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(notdir $(CUDA_SRCS)))

# Main targets
TARGET = $(BIN_DIR)/cellato
TEST_TARGET = $(BIN_DIR)/cellato_tests

ifeq ($(ENABLE_KOKKOS), ON)
DEFINES += -DENABLE_KOKKOS
INCLUDE_DIRS += $(KOKKOS_HOME_INCLUDE:%=-I%)
LIB_DIRS += $(KOKKOS_HOME_LIB:%=-L%)
LIBS += -lkokkoscore $(CUDA_DRIVER_LIB)
EXTRA_OBJS += $(OBJ_DIR)/kokkos_game_of_life_runner.o $(OBJ_DIR)/kokkos_fire_runner.o # $(OBJ_DIR)/kokkos_greenberg_runner.o $(OBJ_DIR)/kokkos_wire_runner.o

$(OBJ_DIR)/kokkos_%_runner.o: $(SRC_DIR)/_relwork/kokkos/kokkos_%_runner.cu | directories
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(DEFINES) $(KOKKOS_HOME_INCLUDE:%=-I%) -extended-lambda

endif

ifeq ($(ENABLE_HALIDE), ON)
DEFINES += -DENABLE_HALIDE
INCLUDE_DIRS += $(HALIDE_HOME_INCLUDE:%=-I%)
LIB_DIRS += $(HALIDE_HOME_LIB:%=-L%)
LIBS += $(HALIDE_LIBS)
EXTRA_OBJS += $(OBJ_DIR)/halide_game_of_life_runner.o # $(OBJ_DIR)/halide_fire_runner.o $(OBJ_DIR)/halide_greenberg_runner.o $(OBJ_DIR)/halide_wire_runner.o

$(OBJ_DIR)/halide_%_runner.o: $(SRC_DIR)/_relwork/halide/halide_%_runner.cpp | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(DEFINES) $(INCLUDE_DIRS)
endif

ifeq ($(ENABLE_GRIDTOOLS), ON)
DEFINES += -DENABLE_GRIDTOOLS
INCLUDE_DIRS += $(GRIDTOOLS_HOME_INCLUDE:%=-I%)
# LIB_DIRS += $(GRIDTOOLS_HOME_LIB:%=-L%)
# LIBS += -lgridtools
EXTRA_OBJS += $(OBJ_DIR)/gridtools_game_of_life_CUDA_runner.o # $(OBJ_DIR)/gridtools_fire_CUDA_runner.o $(OBJ_DIR)/gridtools_greenberg_CUDA_runner.o $(OBJ_DIR)/gridtools_wire_CUDA_runner.o
EXTRA_OBJS += $(OBJ_DIR)/gridtools_game_of_life_CPU_runner.o # $(OBJ_DIR)/gridtools_fire_CPU_runner.o $(OBJ_DIR)/gridtools_greenberg_CPU_runner.o $(OBJ_DIR)/gridtools_wire_CPU_runner.o

$(OBJ_DIR)/gridtools_%_CUDA_runner.o: $(SRC_DIR)/_relwork/gridtools/gridtools_%_runner.cu $(SRC_DIR)/_relwork/gridtools/gridtools_%_runner.inl | directories
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(DEFINES) $(GRIDTOOLS_HOME_INCLUDE:%=-I%)

$(OBJ_DIR)/gridtools_%_CPU_runner.o: $(SRC_DIR)/_relwork/gridtools/gridtools_%_runner.cpp $(SRC_DIR)/_relwork/gridtools/gridtools_%_runner.inl | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(DEFINES) $(GRIDTOOLS_HOME_INCLUDE:%=-I%)
endif

# Rules
all: directories $(TARGET) $(TEST_TARGET)

directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

install_deps: install_kokkos install_halide install_gridtools

install_kokkos: $(VENDOR_DIR)/kokkos
	@echo "Installing Kokkos dependencies..."
	@mkdir -p $(DEP_DIR)/kokkos-build $(DEP_DIR)/kokkos-install
	@$(CMAKE) -B $(DEP_DIR)/kokkos-build -S $(VENDOR_DIR)/kokkos \
		-D CMAKE_BUILD_TYPE=Release \
		-D CMAKE_CXX_COMPILER=$(NVCC_CC_BIN) \
		-D CMAKE_CXX_STANDARD=20 \
		-D CMAKE_CUDA_COMPILER=$(NVCC) \
		-D CMAKE_CUDA_HOST_COMPILER=$(NVCC_CC_BIN) \
		-D Kokkos_ENABLE_CUDA=ON \
		-D Kokkos_ENABLE_CUDA_LAMBDA=ON \
		-D Kokkos_ENABLE_SERIAL=ON \
		-D Kokkos_ENABLE_TESTS=OFF \
		-D Kokkos_ARCH_NATIVE=ON \
		-D Kokkos_ENABLE_DEPRECATED_CODE_4=ON \
		-D CMAKE_INSTALL_PREFIX=$(shell realpath $(DEP_DIR)/kokkos-install)
	@$(CMAKE) --build $(DEP_DIR)/kokkos-build --parallel $(shell nproc) --config Release
	@$(CMAKE) --install $(DEP_DIR)/kokkos-build --prefix=$(shell realpath $(DEP_DIR)/kokkos-install)
	@echo "Kokkos dependencies installed."

# Install LLVM dependencies according to https://github.com/halide/Halide?tab=readme-ov-file#acquiring-llvm
install_llvm: $(VENDOR_DIR)/llvm-project
	@echo "Installing LLVM dependencies..."
	@mkdir -p $(DEP_DIR)/llvm-build $(DEP_DIR)/llvm-install
	@$(CMAKE) -B $(DEP_DIR)/llvm-build -S $(VENDOR_DIR)/llvm-project/llvm \
		-D CMAKE_BUILD_TYPE=Release \
		-D LLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" \
		-D LLVM_ENABLE_RUNTIMES=compiler-rt \
		-D LLVM_TARGETS_TO_BUILD="WebAssembly;X86;AArch64;ARM;Hexagon;NVPTX;PowerPC;RISCV" \
		-D LLVM_ENABLE_ASSERTIONS=ON \
		-D LLVM_ENABLE_EH=ON \
		-D LLVM_ENABLE_RTTI=ON \
		-D LLVM_ENABLE_HTTPLIB=OFF \
		-D LLVM_ENABLE_LIBEDIT=OFF \
		-D LLVM_ENABLE_LIBXML2=OFF \
		-D LLVM_ENABLE_TERMINFO=OFF \
		-D LLVM_ENABLE_ZLIB=OFF \
		-D LLVM_ENABLE_ZSTD=OFF \
		-D LLVM_BUILD_32_BITS=OFF
	@$(CMAKE) --build $(DEP_DIR)/llvm-build --parallel $(shell nproc) --config Release
	@$(CMAKE) --install $(DEP_DIR)/llvm-build --prefix=$(shell realpath $(DEP_DIR)/llvm-install)
	@echo "LLVM dependencies installed."

install_halide: $(VENDOR_DIR)/Halide
	@echo "Installing Halide dependencies..."
	@mkdir -p $(DEP_DIR)/halide $(DEP_DIR)/halide-install
	@$(CMAKE) -B $(DEP_DIR)/halide -S $(VENDOR_DIR)/Halide \
		-D CMAKE_BUILD_TYPE=Release \
		-D Halide_LLVM_ROOT=$(shell realpath $(DEP_DIR)/llvm-install)
	@$(CMAKE) --build $(DEP_DIR)/halide --parallel $(shell nproc) --config Release
	@$(CMAKE) --install $(DEP_DIR)/halide --prefix=$(shell realpath $(DEP_DIR)/halide-install)
	@echo "Halide dependencies installed."

install_gridtools: $(VENDOR_DIR)/gridtools
	@echo "Installing GridTools dependencies..."
	@mkdir -p $(DEP_DIR)/gridtools
	@$(CMAKE) -B $(DEP_DIR)/gridtools -S $(VENDOR_DIR)/gridtools \
		-D CMAKE_BUILD_TYPE=Release \
		-D CMAKE_INSTALL_PREFIX=$(shell realpath $(DEP_DIR)/gridtools)
	@$(CMAKE) --build $(DEP_DIR)/gridtools --parallel $(shell nproc) --config Release
	@$(CMAKE) --install $(DEP_DIR)/gridtools --prefix=$(shell realpath $(DEP_DIR)/gridtools)

print_libraries:
	@echo "${KOKKOS_HOME_LIB} ${HALIDE_HOME_LIB} ${GRIDTOOLS_HOME_LIB}"

# Add directories dependency to object file creation rules
$(MAIN_OBJ): $(MAIN_SRC) | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE_DIRS) $(DEFINES)

$(TEST_OBJ): $(TEST_SRC) | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE_DIRS) $(DEFINES)

# Pattern rule for CUDA files with vpath to find them
vpath %.cu $(CUDA_DIRS)
$(OBJ_DIR)/%.o: %.cu | directories
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(DEFINES)

# Create separate executables for main app and tests
$(TARGET): $(MAIN_OBJ) $(CUDA_OBJS) $(EXTRA_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB_DIRS) $(LIBS)

$(TEST_TARGET): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB_DIRS) $(LIBS)

# Debug rule to print variables
print-%:
	@echo $* = $($*)

run: $(TARGET)
	$(TARGET) $(ARGS)

unit_test: $(TEST_TARGET)
	$(TEST_TARGET) $(ARGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(DEP_DIR)

.PHONY: all run test clean directories print-% install_deps install_kokkos install_llvm install_halide install_gridtools print_libraries
