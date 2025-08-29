#!/bin/sh

set -e

CU_ARCH=90-real

export CXXFLAGS=-w CFLAGS=-w HIPFLAGS=-w CUDAFLAGS=-w

cmake -B build -S . -DGPU_RUNTIME=CUDA -DCMAKE_CUDA_ARCHITECTURES=$CU_ARCH
make -C build -j$(nproc)
