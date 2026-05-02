// Minimal XLA GPU custom-call kernel: element-wise vector add.
//
// Uses API_VERSION_STATUS_RETURNING_UNIFIED (StableHLO api_version = 3).
// XLA invokes this with:
//   void target(CUstream stream, void** buffers,
//               const char* opaque, size_t opaque_len,
//               XlaCustomCallStatus* status)
//
// buffers layout: [in0, in1, ..., out0, out1, ...]
//
// Compile to a shared library with:
//   cd examples/cbits && bash build.sh

#include <cuda_runtime.h>
#include <cuda.h>

// Opaque status struct from XLA.  Success is the default state, so we
// can simply ignore the pointer for successful calls.
typedef struct XlaCustomCallStatus_ XlaCustomCallStatus;

__global__ void vector_add_kernel(const float* a, const float* b,
                                  float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __attribute__((visibility("default")))
void vector_add(CUstream stream, void** buffers,
                const char* opaque, size_t opaque_len,
                XlaCustomCallStatus* status)
{
    const float* a = (const float*)buffers[0];
    const float* b = (const float*)buffers[1];
    float*       c = (float*)      buffers[2];

    // For a demo we assume a fixed size; in production parse from opaque.
    int n = 4;

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vector_add_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
}
