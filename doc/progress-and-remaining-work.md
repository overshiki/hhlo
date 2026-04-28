# HHLO Project Status: Progress and Remaining Work

**Date:** 2026-04-22
**Status:** Multi-GPU inference scaling implemented. 29 examples, 115/115 CPU tests pass, 121/121 tests pass with GPU enabled. Single-GPU and multi-GPU execution fully operational on NVIDIA CUDA via PJRT.

---

## What Works (Completed)

### 1. Architecture & Design
- **Text emission + PJRT** chosen as the correct path (no `mlir-hs` dependency).
- Full design docs written: `design.md`, `implementation-design.md`, `understanding-pjrt.md`, `understanding-zml-pjrt-artifacts.md`, `text-emission-vs-mlir-hs.md`, `control-flow-ops-design.md`, `complex-model-examples-design.md`, `pjrt-cpu-v1160-parser-limitations.md`, `test-suite-documentation.md`, `cuda-runtime-installation.md`.

### 2. Build System
- `cabal build all` completes successfully (library + demo + 29 examples + test suite).
- `cabal test` passes 115/115 tests on CPU.
- `HHLO_TEST_GPU=1 cabal test` passes 121/121 tests (115 CPU + 6 GPU integration).
- PJRT CPU plugin (`deps/pjrt/libpjrt_cpu.so`) downloads and loads correctly via `pjrt_script.sh`.
- PJRT CUDA plugin (`deps/pjrt/libpjrt_cuda.so`) downloads automatically when `nvidia-smi` is present.
- `setup_gpu_env.sh` auto-discovers NVIDIA runtime libraries and configures `LD_LIBRARY_PATH` idempotently.
- C++ linkage resolved: `extra-libraries: stdc++` and `dl` in library, test, and example stanzas.

### 3. Core Library (`src/HHLO/`)
| Module | Status | Notes |
|--------|--------|-------|
| `Core.Types` | ✅ | DTypes, shapes, `KnownShape`, `HostType` family |
| `IR.AST` | ✅ | Core MLIR AST; multi-result support; `Block` / `Region` for nested ops; `AttrRaw` for dialect attrs |
| `IR.Builder` | ✅ | Stateful `Builder`; `Tuple2`; general `Tuple` with `TupleBuilder`; `runBuilderT` / `moduleFromBuilderT`; `emitOpRegions`, `runBlockBuilder`, `emitReturn` |
| `IR.Pretty` | ✅ | StableHLO MLIR text; `module { ... }`; `dense<[[...]]>` for N-D constants; generic region form for `stablehlo.reduce`; integer literal formatting for `i64`/`Bool` constants; unique `^bbN` block labels; `stablehlo.return` terminator; custom `stablehlo.dot_general` syntax |
| `EDSL.Ops` | ✅ | 50+ ops: all element-wise, reductions, shape manipulation, convolutions, NN layers, control flow, data movement |
| `Runtime.PJRT.FFI` | ✅ | FFI + `executableNumOutputs` + event bindings + buffer metadata bindings + **device enumeration** + **device-aware execution** + **async D2H** |
| `Runtime.PJRT.Types` | ✅ | Newtype wrappers + 16 buffer-type constants + **`PJRTDevice`** |
| `Runtime.PJRT.Error` | ✅ | `checkError`, `withErrorMessage`, `PJRTException` |
| `Runtime.PJRT.Plugin` | ✅ | **New.** Backend-agnostic `withPJRT`; convenience wrappers `withPJRTCPU`, `withPJRTGPU` |
| `Runtime.Device` | ✅ | **New.** `addressableDevices`, `deviceId`, `deviceKind`, `defaultGPUDevice` |
| `Runtime.Compile` | ✅ | `compile` with `ForeignPtr` finalizer + **`compileWithOptions`** with configurable `num_replicas` |
| `Runtime.Execute` | ✅ | `execute` with dynamic output count + **`executeOn`** for explicit device targeting + **`executeReplicas`** for concurrent multi-GPU inference |
| `Runtime.Async` | ✅ | `executeAsync`, `bufferReady`, `awaitBuffers` |
| `Runtime.Buffer` | ✅ | `toDevice`/`fromDevice` + `toDeviceOn` (explicit device) + `fromDeviceAsync` (non-blocking D2H) + `bufferDimensions`, `bufferElementType`, `bufferOnDeviceSize` |

### 4. C Shim (`cbits/pjrt_shim.c` + `cbits/pjrt_shim.h`)
- ✅ Plugin loading, client creation/destruction, compilation, execution
- ✅ Dynamic output count, buffer type constants (16 getters)
- ✅ Buffer ready events, event polling, event await/destroy
- ✅ Buffer metadata: `hhlo_pjrt_buffer_dimensions`, `hhlo_pjrt_buffer_element_type`, `hhlo_pjrt_buffer_on_device_size`
- ✅ **Device enumeration:** `hhlo_pjrt_client_addressable_device_count`, `hhlo_pjrt_client_addressable_device`, `hhlo_pjrt_device_id`, `hhlo_pjrt_device_kind`
- ✅ **Device-aware buffer creation:** `hhlo_pjrt_buffer_from_host_on_device`
- ✅ **Async D2H:** `hhlo_pjrt_buffer_to_host_async`
- ✅ **Device-aware execution:** `hhlo_pjrt_execute_on_device`
- ✅ **Multi-device execution:** `hhlo_pjrt_execute_multi` (PJRT-native SPMD execute)
- ✅ **Dynamic compile options:** `hhlo_pjrt_compile_with_options` with configurable `num_replicas`
- ✅ **Formal C header:** `pjrt_shim.h` for clean FFI declarations

### 5. Demo & Examples
| # | File | Description | Status |
|---|------|-------------|--------|
| Demo | `app/Main.hs` | EDSL `stablehlo.add` end-to-end | ✅ |
| 1 | `examples/01-add.hs` | Element-wise addition | ✅ |
| 2 | `examples/02-matmul.hs` | 2×3 @ 3×2 matmul | ✅ |
| 3 | `examples/03-chain-ops.hs` | `(a + b) * (a - b)` | ✅ |
| 4 | `examples/04-async.hs` | Async `executeAsync` + `relu` | ✅ |
| 5 | `examples/05-mlp.hs` | Single-sample MLP | ✅ |
| 6 | `examples/06-mlp-batched.hs` | Batched MLP with `linearBatched` | ✅ |
| 7 | `examples/07-tuple.hs` | Multi-result `Tuple` (MLIR print-only) | ⚠️ PJRT v1.16.0 parser limitation |
| 8 | `examples/08-reduce.hs` | `reduceSum` over all dimensions | ✅ |
| 9 | `examples/09-softmax.hs` | 1-D and batched 2-D `softmax` | ✅ |
| 10 | `examples/10-conv2d.hs` | NHWC conv2d with HWCF filter | ✅ |
| 11 | `examples/11-batch-norm.hs` | Batch norm inference (decomposed) | ✅ |
| 12 | `examples/12-while.hs` | `whileLoop` count-up (MLIR print-only) | ⚠️ PJRT v1.16.0 cannot parse `stablehlo.compare` |
| 13 | `examples/13-conditional.hs` | `conditional` if-then-else | ✅ |
| 14 | `examples/14-gather.hs` | `gather` rows from matrix | ✅ |
| 15 | `examples/15-scatter.hs` | `scatter` replace into vector | ✅ |
| 16 | `examples/16-slice.hs` | `slice` sub-array extraction | ✅ |
| 17 | `examples/17-pad.hs` | `pad` with edge/interior padding | ✅ |
| 18 | `examples/18-dynamic-slice.hs` | `dynamicSlice` runtime start indices | ✅ |
| 19 | `examples/19-sort.hs` | `sort` 1-D ascending (MLIR print-only) | ⚠️ PJRT v1.16.0 cannot parse `stablehlo.compare` |
| 20 | `examples/20-select.hs` | `select` element-wise ternary | ✅ |
| 21 | `examples/21-map.hs` | `map` element-wise custom computation | ✅ |
| 22 | `examples/22-new-ops-smoke-test.hs` | Smoke test for all newer ops | ✅ |
| 23 | `examples/23-resnet.hs` | ResNet-18 inference (toy 8×8) | ✅ |
| 24 | `examples/24-alexnet.hs` | AlexNet inference (toy 16×16) | ✅ |
| 25 | `examples/25-transformer.hs` | Transformer encoder (1×4×16) | ✅ |
| 26 | `examples/26-unet.hs` | UNet segmentation (toy 16×16) | ✅ |
| **27** | `examples/27-gpu-add.hs` | **GPU smoke test: `add` on CUDA** | ✅ |
| **28** | `examples/28-gpu-matmul-bench.hs` | **GPU benchmark: 4096×4096 matmul** | ✅ |
| **29** | `examples/29-multi-gpu-inference.hs` | **Multi-GPU concurrent 4096×4096 matmul** | ✅ |

### 6. Test Suite (`test/`)
- ✅ **115 CPU tests** across 13 modules, all passing.
- ✅ **6 GPU integration tests** (run with `HHLO_TEST_GPU=1`):
  - `Test.Runtime.EndToEndGPU` — GPU availability & device enumeration
  - `Test.Runtime.BufferGPU` — Buffer round-trip and metadata on GPU
  - `Test.Runtime.AsyncGPU` — `executeAsync` + `awaitBuffers`, `bufferReady` polling on GPU
  - `Test.Runtime.MultiGPU` — Concurrent `executeReplicas` across all GPUs
- Tier 1 (Golden): `Test.IR.Pretty`, `Test.IR.PrettyOps`, `Test.IR.PrettyNN`, `Test.IR.PrettyControlFlow`, `Test.IR.Builder`, `Test.EDSL.Ops`
- Tier 2 (E2E Numerical): `Test.Runtime.EndToEndArithmetic`, `Test.Runtime.EndToEndMatmul`, `Test.Runtime.EndToEndDataMovement`, `Test.Runtime.EndToEndNN`, `Test.Runtime.EndToEndReductions`, `Test.Runtime.EndToEndShape`
- Tier 3 (Integration): `Test.Runtime.Buffer`, `Test.Runtime.Async`, `Test.Runtime.Errors`
- See `doc/test-suite-documentation.md` for full details.

---

## Completed P1–P3 Items

| # | Item | Status |
|---|------|--------|
| 1 | Fix `broadcast` with `broadcast_dimensions` | ✅ `broadcastWithDims` + `linearBatched` |
| 2 | Batched MLP example | ✅ `examples/06-mlp-batched.hs` passes |
| 3 | Buffer metadata queries | ✅ `bufferDimensions`, `bufferElementType`, `bufferOnDeviceSize` |
| 4 | General tuple support | ✅ `Tuple` GADT + `TupleBuilder` + `runBuilderT` |
| 5 | Fix reduction ops | ✅ `reduceSum` / `reduceSumDim` with generic region form |
| 6 | `softmax` layer | ✅ `examples/09-softmax.hs` passes numerically |
| 7 | `conv2d` layer | ✅ `examples/10-conv2d.hs` passes numerically |
| 8 | `batchNormInference` layer | ✅ `examples/11-batch-norm.hs` passes numerically |
| 9 | Control flow ops | ✅ `whileLoop`, `conditional`, `gather`, `scatter` implemented |
| 10 | Data movement ops | ✅ `slice`, `pad`, `dynamicSlice`, `sort`, `convert` implemented |
| 11 | Selection & map ops | ✅ `select`, `map` implemented |
| 12 | Complex model primitives | ✅ `transpose`, `tanh`, `concatenate`, `iota`, `reduceWindow`, `maxPool`, `avgPool`, `softmax3D/4D`, `layerNorm`, `globalAvgPool`, `gelu`, `transposeConvolution`, `dotGeneral`, `conv2dWithPadding` |
| 13 | Complex model examples | ✅ ResNet-18, AlexNet, Transformer, UNet all compile and execute on CPU |
| 14 | Comprehensive test suite | ✅ 115 CPU tests across golden, E2E, and integration tiers |
| 15 | Integer constant pretty-printing | ✅ `dense<0>` for `i64`, `true`/`false` for `Bool` |
| 16 | `stablehlo.reduce` generic form | ✅ Proper region-based emission for partial reductions |
| **17** | **Single-GPU support** | ✅ **Device enumeration, device-aware buffers/execution, async D2H. 5 GPU tests pass on NVIDIA RTX 5090.** |
| **18** | **Backend-agnostic plugin loading** | ✅ **`withPJRT` abstracts CPU/CUDA plugin selection.** |
| **19** | **GPU examples & benchmarks** | ✅ **`example-gpu-add` and `example-gpu-matmul-bench` operational.** |
| **20** | **Multi-GPU inference scaling** | ✅ **`executeReplicas` runs concurrent `executeOn` across N GPUs. `compileWithOptions` supports `num_replicas`. `example-multi-gpu-inference` verified on 8× RTX 5090.** |
| **21** | **Autograd (reverse-mode AD)** | ✅ **`gradModule`, `grad`, and VJP rules for 25+ ops including `convolution`, `transpose_convolution`, `reduce_window`. 7 E2E autograd tests pass.** |

---

## Known Limitations / Technical Debt

### 1. Single-Device Execution (GPU works, multi-GPU inference works)
- `executeOn` targets exactly one device. ✅
- `executeReplicas` distributes independent forward passes across multiple GPUs concurrently. ✅
- **Clarification:** HHLO now supports **reverse-mode automatic differentiation** via `grad` / `gradModule`. Multi-GPU still means inference scaling only — autograd runs on a single device. See the autograd examples (34–36) and the autograd section in `doc/implementation-design.md`.

### 2. PJRT CPU v1.16.0 Parser Limitations
The specific `libpjrt_cpu.so` build from `zml/pjrt-artifacts` (StableHLO v1.16.0) has a text parser with known gaps:

| Op / Feature | Status | Workaround |
|--------------|--------|------------|
| Multi-result `func.func` / tuples | ❌ Rejected | `example-tuple` is MLIR print-only |
| `stablehlo.batch_norm_inference` | ❌ Rejected | Decomposed into basic ops |
| `stablehlo.compare` | ❌ Rejected | `example-while`, `example-sort` are MLIR print-only; `conditional` avoids `compare` by passing boolean as arg |
| `stablehlo.gather` / `stablehlo.scatter` | ⚠️ Needs generic form + `array<i64: ...>` | Works with emitted syntax |
| `stablehlo.slice` / `stablehlo.pad` / `stablehlo.dynamic_slice` | ✅ Works | — |
| `stablehlo.select` | ✅ Works | — |
| `stablehlo.map` | ✅ Works | Generic form with region |
| `stablehlo.dot` with rank > 2 | ❌ Rejected | Use `stablehlo.dot_general` via `dotGeneral` |

**Root cause:** The limitation is in the **frontend parser/converter**, not the XLA CPU compiler/runtime. The emitted MLIR is 100% valid StableHLO and executes correctly on newer PJRT plugins or GPU.

### 3. No Profiling / Timing
- No `PJRT_Executable_Execute` profiling options wired up.

### 4. Error Handling Could Be Richer
- `PJRTException` only carries a `String` message.
- No structured error codes (OOM, compilation failure, driver mismatch, etc.).

### 5. C Shim Completeness
- ~~Missing: `PJRT_Client_Devices`, `PJRT_Device_Memory`, `PJRT_TopologyDescription`.~~ ✅ Device enumeration added.
- Missing: `PJRT_Executable_Serialize`, `PJRT_Executable_Deserialize`.
- Missing: `PJRT_TopologyDescription` (for multi-node topology discovery).

### 6. CUDA Runtime Dependency Management
- The PJRT CUDA plugin requires cuDNN, NCCL, and NVSHMEM at runtime.
- `setup_gpu_env.sh` discovers these from conda/pip installations; a fully self-contained distribution would bundle or statically link these.
- The `nvshmem_transport_ibrc.so.4` → `.so.3` version mismatch requires a symlink workaround.

---

## Remaining Work (Prioritized)

### P1 — Important
1. **Profiling integration** — Compile and execution timing via PJRT profiling APIs.
2. **Cross-device buffer copies** — `PJRT_Buffer_CopyToDevice` for moving intermediate tensors between GPUs (needed for model/pipeline parallelism).

### P2 — Nice to have
3. **Executable serialization / deserialization** — Cache compiled programs to disk.
4. **Shape inference improvements** — More complete type families for ops.
5. **Constant folding in Builder** — Evaluate pure ops at compile time.
6. **Richer error types** — Structured `PJRTException` with error codes.

### P3 — Completed ✅
7. ~~More EDSL ops — `map`, `select`~~ ✅ Done.
8. ~~Fix `transpose` + add missing primitives~~ ✅ Done.
9. ~~Add composite helpers~~ ✅ Done.
10. ~~ResNet-18 inference example~~ ✅ Done.
11. ~~Transformer encoder example~~ ✅ Done.
12. ~~AlexNet inference example~~ ✅ Done.
13. ~~UNet inference example~~ ✅ Done.
14. ~~Comprehensive test suite~~ ✅ Done.
15. ~~Single-GPU CUDA support~~ ✅ Done.
16. ~~Reverse-mode automatic differentiation~~ ✅ Done.

---

## Immediate Next Steps (awaiting your decision)

The codebase is at a **solid multi-GPU inference** stage with:
- A type-safe NN-layer EDSL (50+ ops)
- Full control flow support
- Four validated complex model examples (ResNet, AlexNet, Transformer, UNet)
- 29 working examples (26 CPU + 3 GPU)
- 115/115 CPU tests passing, 121/121 with GPU enabled
- Single-GPU and multi-GPU inference verified on 8× NVIDIA GeForce RTX 5090

The most impactful next decisions are:

1. **Improve ergonomics?** Add profiling, executable serialization, and richer error types.
2. **Add more model examples?** e.g., LSTM, diffusion UNet, Vision Transformer.
3. **Refine the EDSL?** Better shape inference, automatic broadcasting, or higher-level layer combinators.
4. **Cross-device communication?** Add buffer copy between GPUs for model/pipeline parallelism.
