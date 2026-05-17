# HHLO — Haskell Frontend for StableHLO

HHLO is a Haskell library for building, compiling, and executing machine learning programs that target [StableHLO](https://github.com/openxla/stablehlo), the portable, versioned IR of the [OpenXLA](https://openxla.org/) ecosystem.

It lets you write ML models in pure Haskell with compile-time shape checking, compile them to CPU or GPU via the [PJRT](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) C API, and even differentiate them automatically — all without leaving the type system.

```haskell
{-# LANGUAGE DataKinds, TypeApplications #-}
import HHLO.Session
import HHLO.EDSL.Ops
import HHLO.Autograd

-- Define a model, differentiate it, and run it on CPU in 6 lines.
main = withCPU $ \sess -> do
    let f x = sumAll =<< multiply x x
        gradMod = gradModule @'[3] @'F32 f
    compiled <- compile sess gradMod
    result   <- run sess compiled (hostFromList @'[3] @'F32 [1, 2, 3])
    print (hostToList result)   -- [2.0, 4.0, 6.0]
```

---

## Table of Contents

- [Why HHLO?](#why-hhlo)
- [Design Philosophy](#design-philosophy)
- [Features](#features)
  - [Type-Safe EDSL](#type-safe-edsl)
  - [Convenience Ops](#convenience-ops)
  - [Autograd](#autograd)
  - [Runtime & Hardware](#runtime--hardware)
  - [Control Flow & RNG](#control-flow--rng)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [License](#license)

---

## Why HHLO?

Most ML frameworks trace Python code to build computation graphs. HHLO takes a different path: you write StableHLO directly in Haskell.

This means:

- **No Python runtime** — Your model is ordinary Haskell code.
- **Compile-time shape safety** — Matmul mismatches are type errors, not runtime failures.
- **Native autograd** — Reverse-mode differentiation is implemented as a Haskell library, not a C++ backend.
- **True portability** — StableHLO is a standardized, versioned IR; the same Haskell code runs on CPU, NVIDIA GPU, or any future PJRT backend.

---

## Design Philosophy

### Text Emission + PJRT

HHLO emits StableHLO MLIR text and hands it straight to `PJRT_Client_Compile`. This is the same compilation path used by JAX's C++ backend, but without the heavy dependency of building LLVM/MLIR from source.

### Phantom Types

Every tensor carries its shape and dtype as phantom type parameters:

```haskell
Tensor '[2, 3] 'F32   -- 2×3 matrix of Float32
Tensor '[4]    'F64   -- 4-element vector of Float64
```

Matmul, broadcast, and conv shapes are checked at compile time via type families. If the shapes don't match, GHC tells you before you ever load a PJRT plugin.

### Layered Architecture

HHLO is structured so you can use as much or as little abstraction as you need:

```
┌─────────────────────────────────────┐
│  Session (HHLO.Session)             │  One-liners: withCPU, compile, run
├─────────────────────────────────────┤
│  Autograd (HHLO.Autograd)           │  grad, vjp, gradModule — reverse-mode AD
├─────────────────────────────────────┤
│  EDSL (HHLO.EDSL.Ops)               │  Type-safe frontend: add, matmul, einsum, etc.
├─────────────────────────────────────┤
│  IR Builder (HHLO.IR.Builder)       │  Stateful monad for constructing MLIR
├─────────────────────────────────────┤
│  Pretty Printer (HHLO.IR.Pretty)    │  Emits StableHLO MLIR text
├─────────────────────────────────────┤
│  PJRT Runtime (HHLO.Runtime.*)      │  Compile → Execute on CPU or GPU
└─────────────────────────────────────┘
```

The high-level layers (`Session`, `Autograd`) eliminate PJRT boilerplate for the common case. The low-level layers (`IR.Builder`, `Pretty`, `Runtime`) remain available when you need full control.

---

## Features

### Type-Safe EDSL

The frontend provides 50+ typed ops covering arithmetic, linear algebra, reductions, data movement, and neural network primitives:

```haskell
-- Arithmetic
c <- add a b
d <- multiply a b
e <- matmul a b

-- Non-linear
y <- relu x
y <- sigmoid x
y <- softmax x

-- Reductions
s <- sumAll x                    -- reduce all dims → scalar
v <- reduceSumDim @0 x           -- reduce dim 0

-- Data movement
sliced <- slice x [(0, 2), (1, 3)]   -- extract sub-array
padded <- pad x 0 [(1, 1), (0, 0)]   -- pad with zeros
trans  <- transpose x [1, 0]         -- permute dimensions
```

Shape mismatches are caught at compile time. A matmul between a `[2,3]` and a `[4,5]` tensor is a type error, not a segfault.

### Convenience Ops

Beyond the raw StableHLO surface, HHLO provides higher-level combinators that compose primitive ops into familiar patterns:

| Op | What it does |
|---|---|
| `einsum "ij,jk->ik" a b` | Einstein summation (dispatches to `dotGeneral` + `transpose`) |
| `split dim n t` | Decompose a tensor into `n` equal slices along `dim` |
| `stack dim [t1, t2, ...]` | Concatenate tensors along a new axis `dim` |
| `productAll t` | Reduce all dimensions with `multiply` (like `sumAll` but product) |
| `productDim dims t` | Reduce specific dimensions with `multiply` |
| `topK k t` | Return the top-K elements (sort descending + slice) |

These are implemented as pure compositions of existing EDSL ops, so they inherit full autograd support automatically.

### Autograd

HHLO includes a native reverse-mode automatic differentiation engine that transforms StableHLO computation graphs into their gradients.

**Standalone modules** — produce a reusable `Module`:

```haskell
-- f(x) = sum(x²)   =>   grad f(x) = 2x
gradMod :: Module
gradMod = gradModule @'[3] @'F32 $ \x -> do
    sq <- multiply x x
    sumAll sq
```

**Multi-parameter gradients** — differentiate w.r.t. multiple inputs natively:

```haskell
-- g(x, y) = sum(x * y)   =>   (grad_x = y, grad_y = x)
(gradX, gradY) <- grad2 (\x y -> sumAll =<< multiply x y) xVal yVal
```

**Structured parameters with ParamTree** — train models with many weights
without manual pack/slice bookkeeping:

```haskell
{-# LANGUAGE DeriveGeneric #-}

data MLPParams = MLPParams
    { w1 :: Tensor '[2,2] 'F32
    , b1 :: Tensor '[2]   'F32
    , w2 :: Tensor '[1,2] 'F32
    , b2 :: Tensor '[1]   'F32
    } deriving (Generic)

instance ParamTree MLPParams

loss p x = do
    h  <- relu =<< add (matmul x (w1 p)) (b1 p)
    y  <- add (matmul h (w2 p)) (b2 p)
    diff <- sub y target
    sumAll =<< multiply diff diff

-- Returns an MLPParams of gradients
dParams <- gradWithParams loss params x
```

**In-place combinators** — use inside `buildModule` for composability:

```haskell
buildModule @1 @1 "loss_and_grad" $ \x -> do
    loss <- sumAll =<< multiply x x
    g    <- grad (\y -> sumAll (multiply y y)) x
    returnTuple2 loss g
```

**Vector-Jacobian products** — for non-scalar outputs:

```haskell
-- vjp f x seed = (Df(x))ᵀ · seed
vjpModule @'[3] @'[2] @'F32
    (\x -> do w <- constant @'[2,3] @'F32 1.0; matmul w x)
```

Supported ops: `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `log`, `sqrt`, `power`, `sine`, `cosine`, `tanh`, `abs`, `maximum`, `minimum`, `reshape`, `transpose`, `broadcast_in_dim`, `reduce` (sum), `dot`, `select`, `slice`, `pad`, `concatenate`, `convert`, `convolution`, `reduce_window`, and more.

Ops without gradient rules (e.g. `compare`, `floor`, `ceil`, `sort`) safely return zero gradients. Stubs (e.g. `gather`, `scatter`) error explicitly.

### Runtime & Hardware

**CPU & GPU**

The same Haskell code compiles to CPU via `withCPU` or to GPU via `withGPU`:

```haskell
withCPU $ \sess -> do ...   -- CPU plugin, works out of the box
withGPU $ \sess -> do ...   -- CUDA plugin, requires NVIDIA runtime libs
```

**Async Execution**

`HHLO.Runtime.Async` provides true non-blocking execution:

```haskell
bufs <- executeAsync api exec inputs
ready <- bufferReady api (head bufs)   -- poll
awaitBuffers api bufs                   -- block until done
```

**Multi-GPU Inference**

Run the same compiled model concurrently across multiple GPUs:

```haskell
compileWithOptions api client mlirText
    (defaultCompileOptions { optNumReplicas = numDevs })

executeReplicas api exec
    [ (gpu0, [bufA0, bufB0])
    , (gpu1, [bufA1, bufB1])
    , ...
    ]
```

**ForeignPtr Finalizers**

PJRT buffers and executables are managed by `ForeignPtr` finalizers. They are automatically destroyed when garbage-collected — no explicit cleanup required.

### Control Flow & RNG

**Multi-Value Control Flow**

`whileLoop2` / `conditional2` carry multiple typed tensors through loops and conditionals without manual packing:

```haskell
(resultCounter, resultSum) <- whileLoop2 counter0 sum0
    (\c s -> compare c limit "LT")
    (\c s -> do
        cNext <- add c one
        sNext <- add s cNext
        returnTuple2 cNext sNext)
```

**Random Number Generation**

```haskell
uniform  <- rngUniform a b      -- uniform in [a, b)
normal   <- rngNormal            -- standard normal (mean 0, std 1)
(newSt, bits) <- rngBitGenerator state   -- Threefry bit generator
```

---

## Quick Start

### 1. Download PJRT plugins

```bash
./pjrt_script.sh
```

This fetches `libpjrt_cpu.so` into `deps/pjrt/`. If you have an NVIDIA GPU, the CUDA plugin is also downloaded automatically.

You can also point HHLO to an existing PJRT plugin via environment variables:
```bash
export HHLO_PJRT_CPU_PLUGIN=/path/to/libpjrt_cpu.so
export HHLO_PJRT_GPU_PLUGIN=/path/to/libpjrt_cuda.so
```

### 2. Build

```bash
cabal build all
```

### 3. Run an example

```bash
# CPU — works out of the box
cabal run example-add --flag=examples

# Autograd
cabal run example-autograd-basic --flag=examples
```

### 4. Run tests

```bash
cabal test                    # 191 CPU tests
```

GPU tests are **opt-in** via the `HHLO_TEST_GPU` environment variable (they require an NVIDIA GPU and the PJRT CUDA plugin):

```bash
HHLO_TEST_GPU=1 cabal test    # 191 CPU + 82 GPU tests = 273 total
```

---

## Examples

Standalone examples live in `examples/` and cover arithmetic, neural networks, control flow, RNG, and autograd:

| # | Command | Description |
|---|---------|-------------|
| 1 | `example-add` | Element-wise `c = a + b` |
| 2 | `example-matmul` | 2×3 @ 3×2 matrix multiply |
| 3 | `example-chain-ops` | `(a + b) * (a - b)` |
| 4 | `example-async` | Async `executeAsync` + `relu` |
| 5 | `example-mlp` | 2-layer MLP |
| 6 | `example-mlp-batched` | Batched MLP |
| 7 | `example-tuple` | Multi-result `func.func` |
| 8 | `example-reduce` | `reduceSum` over all dimensions |
| 9 | `example-softmax` | 1-D and batched 2-D softmax |
| 10 | `example-conv2d` | NHWC conv2d |
| 11 | `example-batch-norm` | Batch norm inference |
| 12 | `example-while` | `whileLoop` count-up |
| 13 | `example-conditional` | `conditional` if-then-else |
| 14 | `example-gather` | `gather` rows from matrix |
| 15 | `example-scatter` | `scatter` replace into vector |
| 16 | `example-slice` | `slice` sub-array extraction |
| 17 | `example-pad` | `pad` with edge/interior padding |
| 18 | `example-dynamic-slice` | `dynamicSlice` runtime indices |
| 19 | `example-sort` | `sort` 1-D ascending |
| 20 | `example-select` | Element-wise ternary `select` |
| 21 | `example-map` | `map` with custom computation |
| 22 | `example-new-ops-smoke-test` | Smoke test for newer ops |
| 23 | `example-resnet` | ResNet-18 toy (8×8 input) |
| 24 | `example-alexnet` | AlexNet toy (16×16 input) |
| 25 | `example-transformer` | Transformer encoder (1×4×16) |
| 26 | `example-unet` | UNet segmentation toy (16×16) |
| 30 | `example-rng-uniform` | `rngUniform` random floats [0,1) |
| 31 | `example-rng-normal` | `rngNormal` standard normal distribution |
| 32 | `example-rng-bit-generator` | `rngBitGenerator` Threefry PRNG |
| 33 | `example-multi-value-loop` | `whileLoop2` with two loop-carried values |
| **34** | **`example-autograd-basic`** | **Gradient of `sum(x²)`** |
| **35** | **`example-autograd-linear`** | **Gradient of linear + MSE loss** |
| **36** | **`example-autograd-composite`** | **Gradient through ReLU + linear + sum** |
| **37** | **`example-autograd-multiparam`** | **`gradWithParams` on a record of weights** |
| 27 | `example-gpu-add` | GPU smoke test |
| 28 | `example-gpu-matmul-bench` | GPU 4096×4096 benchmark |
| 29 | `example-multi-gpu-inference` | Multi-GPU concurrent matmul |

> **Note:** All `example-*` executables are guarded by the `examples` flag in `hhlo.cabal` (defaults to `False`). Append `--flag=examples` to every `cabal run example-*` command.

### Writing your own model

```haskell
{-# LANGUAGE DataKinds, TypeApplications #-}

import HHLO.Session
import HHLO.EDSL.Ops
import HHLO.Autograd

-- A tiny model: predict y from x via a learned weight.
-- We want the gradient of the squared error.
main = withCPU $ \sess -> do
    let model x = do
            w <- constant @'[1] @'F32 2.0   -- fixed weight for demo
            b <- constant @'[1] @'F32 1.0
            y <- add =<< multiply w x =<< pure b
            tgt <- constant @'[1] @'F32 5.0
            diff <- sub y tgt
            sumAll =<< multiply diff diff

    let gradMod = gradModule @'[1] @'F32 model
    compiled <- compile sess gradMod
    result   <- run sess compiled (hostFromList @'[1] @'F32 [3.0])
    print (hostToList result)   -- [8.0]
```

---

## Installation

### System Requirements

- GHC 9.6+ and Cabal 3.10+
- Linux x86_64 (other platforms supported by PJRT artifacts may work)
- `curl`, `tar`, and standard C toolchain (`gcc` or `clang`)
- `libstdc++` and `libdl` (usually present on Linux)

### From Hackage

```cabal
build-depends: hhlo >= 0.5
```

Or:

```bash
cabal install hhlo
```

### GPU Setup

The PJRT CUDA plugin depends on **cuDNN**, **NCCL**, and **NVSHMEM**. If you already have them (e.g. via PyTorch or JAX):

```bash
./setup_gpu_env.sh
source ~/.bashrc
```

This auto-discovers the libraries and appends them to `~/.bashrc`. After that, GPU examples work directly:

```bash
cabal run example-gpu-add --flag=examples
cabal run example-gpu-matmul-bench --flag=examples
```

---

## License

MIT License — see [LICENSE](LICENSE).
