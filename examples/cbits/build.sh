#!/bin/bash
# Build script for the vector_add custom-call example.
# Requires nvcc in PATH.

set -e

NVCC=${NVCC:-nvcc}

# Auto-detect GPU architecture if not explicitly set
if [ -z "$ARCH" ]; then
    if command -v nvidia-smi &> /dev/null; then
        # Try to get compute capability from nvidia-smi
        CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
        if [ -n "$CAP" ]; then
            ARCH="sm_${CAP}"
            echo "Auto-detected GPU architecture: ${ARCH}"
        fi
    fi
    # Fallback default
    if [ -z "$ARCH" ]; then
        ARCH=sm_80
        echo "Could not auto-detect architecture, falling back to ${ARCH}"
    fi
fi

OUTDIR="$(dirname "$0")"
cd "$OUTDIR"

$NVCC -shared -o libvector_add.so \
    -Xcompiler -fPIC \
    -gencode arch=compute_${ARCH#sm_},code=$ARCH \
    vector_add.cu \
    -lcudart -O3

echo "Built $OUTDIR/libvector_add.so for $ARCH"
