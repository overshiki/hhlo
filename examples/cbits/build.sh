#!/bin/bash
# Build script for the vector_add custom-call example.
# Requires nvcc in PATH.

set -e

NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_80}

OUTDIR="$(dirname "$0")"
cd "$OUTDIR"

$NVCC -shared -o libvector_add.so \
    -Xcompiler -fPIC \
    -gencode arch=compute_${ARCH#sm_},code=$ARCH \
    vector_add.cu \
    -lcudart -O3

echo "Built $OUTDIR/libvector_add.so"
