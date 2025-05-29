CC=g++
NVCC=nvcc
NVCC_CC_BIN = $(shell which $(CC))
CXXFLAGS= -O3 -std=c++11 -lm -DNDEBUG
CUDAFLAGS= -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_90,code=sm_90 \
           -std=c++11 -c
LIBS= -lcudart

EXE=main

all: $(EXE)

GOL_AN5D_2048_host.o : GOL_AN5D_2048_host.cu
	$(NVCC) -ccbin $(NVCC_CC_BIN) $(CXXFLAGS) $(CUDAFLAGS) GOL_AN5D_2048_host.cu 

GOL_AN5D_2048_kernel.o: GOL_AN5D_2048_kernel.cu 
	$(NVCC) -ccbin $(NVCC_CC_BIN) $(CXXFLAGS) $(CUDAFLAGS) GOL_AN5D_2048_kernel.cu  

NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
LDLIBS += -L$(CUDA_PATH)/lib64 -lcudart

main: GOL_AN5D_2048_host.o GOL_AN5D_2048_kernel.o
	$(NVCC) -ccbin $(NVCC_CC_BIN) $(CXXFLAGS) -o GOL_an5d GOL_AN5D_2048_host.o GOL_AN5D_2048_kernel.o -O3 -Xcompiler -Wextra -std=c++11 -lm $(LDLIBS)

clean:
	rm -rf GOL_an5d *.o

# TO TAKE INTO ACCOUNT:
# 1) The real architecture must be an implementation of the virtual architecture
# 2) The virtual architecture should always be chosen as low as possible, thereby maximizing the actual GPUs to run on. 
# 3) The real architecture should be chosen as high as possible 
#
# GPU REAL Architecture Feature List:
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
# 
# 	                        Basic features
# sm_30 and sm_32           + Kepler support
#                           + Unified memory programming
# sm_35	                    + Dynamic parallelism support
# sm_50, sm_52, and sm_53	+ Maxwell support
# sm_60, sm_61, and sm_62	+ Pascal support
# sm_70 and sm_72	        + Volta support
# sm_75	                    + Turing support
#
#
# GPU VIRTUAL Architecture Feature List:
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
#
# 	                                        Basic features
# compute_30 and compute_32                 + Kepler support
#                                           + Unified memory programming
# compute_35	                            + Dynamic parallelism support
# compute_50, compute_52, and compute_53	+ Maxwell support
# compute_60, compute_61, and compute_62	+ Pascal support
# compute_70 and compute_72	                + Volta support
# compute_75	                            + Turing support

