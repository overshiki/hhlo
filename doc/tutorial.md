# HHLO: A Complete Tutorial

> **From `add` to distributed GPU training — a guided tour of Haskell's StableHLO frontend.**
>
> *This document assumes basic familiarity with Haskell (types, monads, type families) and elementary linear algebra. No prior ML framework experience is required.*

---

## Table of Contents

1. [What is HHLO?](#1-what-is-hhlo)
2. [Your First Program](#2-your-first-program)
3. [Shapes as Types](#3-shapes-as-types)
4. [The EDSL in Depth](#4-the-edsl-in-depth)
5. [Building and Executing](#5-building-and-executing)
6. [Neural Network Primitives](#6-neural-network-primitives)
7. [Automatic Differentiation](#7-automatic-differentiation)
8. [Control Flow](#8-control-flow)
9. [Multi-Device Execution](#9-multi-device-execution)
10. [How It Works](#10-how-it-works)
11. [Appendix: Quick Reference](#11-appendix-quick-reference)

---

## 1. What is HHLO?

HHLO is a Haskell library that lets you write machine-learning models in pure Haskell, compile them to [StableHLO](https://github.com/openxla/stablehlo) (a portable, versioned ML IR), and execute them on CPU or GPU via the [PJRT](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) C API.

If you've used JAX or PyTorch, think of HHLO as **"JAX without Python"** — or more precisely, as a way to write XLA-compatible programs directly in a strongly-typed functional language. The key ideas are:

- **Compile-time shape safety**: A matmul between a `[2,3]` and a `[4,5]` tensor is a *type error*, not a runtime crash.
- **Native autograd**: Reverse-mode differentiation is implemented entirely in Haskell, not a C++ backend.
- **True portability**: StableHLO is a standardized IR; the same Haskell code runs on CPU, NVIDIA GPU, or any future PJRT backend without recompilation.
- **Zero Python runtime**: Your model is ordinary Haskell code. No tracing, no graph construction, no GIL.

---

## 2. Your First Program

Let's start with the smallest possible HHLO program: adding two numbers.

### 2.1 Installation

First, download the PJRT CPU plugin:

```bash
./pjrt_script.sh        # Fetches libpjrt_cpu.so into deps/pjrt/
```

Then build:

```bash
cabal build all
```

### 2.2 Hello, Addition

```haskell
{-# LANGUAGE DataKinds, TypeApplications #-}

import HHLO.Session
import HHLO.EDSL.Ops

main :: IO ()
main = withCPU $ \sess -> do
    -- Build a tiny model: c = a + b
    let model = buildModule @2 @1 "add" $ \a b -> add a b

    compiled <- compile sess model

    -- Create input tensors on the host
    aHost <- hostFromList @'[1] @'F32 [3.0]
    bHost <- hostFromList @'[1] @'F32 [4.0]

    -- Run on CPU
    [cHost] <- run sess compiled [aHost, bHost]

    print (hostToList cHost)   -- [7.0]
```

Let's unpack this:

- `buildModule @2 @1 "add"` creates a module with **2 inputs** and **1 output**. The `@2` and `@1` are type-level naturals.
- `\a b -> add a b` is the model logic. `a` and `b` are `Tensor '[1] 'F32` — 1-element vectors of Float32.
- `withCPU` handles plugin loading, client creation, and cleanup.
- `hostFromList` and `hostToList` convert between Haskell lists and HHLO's typed host tensors.

### 2.3 The Session API

The `Session` API is the highest-level entry point. It manages the entire lifecycle:

```haskell
withCPU  :: (Session -> IO a) -> IO a   -- CPU plugin
withGPU  :: (Session -> IO a) -> IO a   -- Auto-detects first GPU
withGPUDevice :: Int -> (Session -> IO a) -> IO a  -- Specific GPU by index
```

A `Session` gives you `compile` and `run`:

```haskell
compile :: Session -> Module -> IO CompiledModel
run     :: Session -> CompiledModel -> [HostTensor] -> IO [HostTensor]
```

If you prefer lower-level control (explicit device targeting, async execution, multi-GPU), the `HHLO.Runtime.*` modules are always available.

---

## 3. Shapes as Types

The most distinctive feature of HHLO is that tensor shapes live in the type system.

### 3.1 Phantom Types

```haskell
Tensor '[2, 3] 'F32   -- 2×3 matrix of Float32
Tensor '[4]    'F64   -- 4-element vector of Float64
Tensor '[]     'F32   -- scalar (empty shape)
```

`'[2, 3]` is a type-level list of naturals. `'F32` is a type-level datatype. These are **phantom types**: they carry no runtime data, but GHC checks them at compile time.

### 3.2 Why This Matters

```haskell
-- This compiles:
let a = undefined :: Tensor '[2, 3] 'F32
let b = undefined :: Tensor '[3, 4] 'F32
matmul a b   -- Tensor '[2, 4] 'F32

-- This is a COMPILE ERROR:
let c = undefined :: Tensor '[2, 3] 'F32
let d = undefined :: Tensor '[4, 5] 'F32
matmul c d   -- Type error! Inner dimensions don't match.
```

The error comes from the `MatMulShape` type family:

```haskell
type family MatMulShape (a :: Shape) (b :: Shape) :: Shape where
    MatMulShape '[m, n] '[n, p] = '[m, p]
    -- No other instances = type error for mismatched shapes
```

### 3.3 Type-Level Programming Primer

You don't need to be a type-level wizard to use HHLO, but understanding a few patterns helps:

| Concept | What it means | Example |
|---------|--------------|---------|
| `KnownShape s` | Constraint that shape `s` can be read at runtime | `shapeVal (Proxy @'[2,3]) == [2,3]` |
| `KnownDType d` | Constraint that dtype `d` can be read at runtime | `dtypeVal (Proxy @'F32) == F32` |
| `Proxy` | A value-level witness for a type | `Proxy @'[2,3] :: Proxy '[2,3]` |
| `TypeApplications` | Syntax to pass types explicitly | `constant @'[2,2] @'F32 1.0` |

These constraints are automatically satisfied when you use concrete types like `'[2,3]` and `'F32`. GHC handles the proof.

---

## 4. The EDSL in Depth

The EDSL (`HHLO.EDSL.Ops`) provides 50+ typed operations. They fall into several categories.

### 4.1 Element-wise Arithmetic

```haskell
c <- add a b
d <- subtract a b
e <- multiply a b
f <- divide a b
g <- negate a
h <- abs a
```

All of these require operands of the same shape and dtype. The result has the same shape.

### 4.2 Non-linearities

```haskell
y <- relu x              -- max(x, 0)
y <- sigmoid x           -- 1 / (1 + exp(-x))
y <- tanh x
y <- softmax x           -- Softmax over the last axis
y <- gelu x              -- Gaussian Error Linear Unit
```

### 4.3 Reductions

```haskell
s <- sumAll x                    -- Sum all elements → scalar
s <- productAll x                -- Product of all elements → scalar
v <- reduceSumDim @0 x           -- Reduce dimension 0
v <- reduceSumDim @1 x           -- Reduce dimension 1
v <- reduceMeanDim @0 x          -- Mean along dimension 0
```

The `@0`, `@1` are type-level naturals specifying which dimension to reduce.

### 4.4 Linear Algebra

```haskell
-- Matrix multiply: [m,n] × [n,p] → [m,p]
c <- matmul a b

-- General dot product with custom contracting/batching dims
c <- dotGeneral a b
    (DotDimensionNums [1] [0] [] [])   -- contract a's dim 1 with b's dim 0

-- Einstein summation
c <- einsum "ij,jk->ik" a b         -- Same as matmul
c <- einsum "ii->i" a               -- Diagonal extraction
c <- einsum "ij->ji" a              -- Transpose
c <- einsum "bij,bjk->bik" a b      -- Batched matmul
```

`einsum` is a convenience wrapper around `dotGeneral` + `transpose`. It parses the subscript string, computes the required dimension numbers, and emits the correct ops.

### 4.5 Shape Manipulation

```haskell
-- Reshape: change dimensions while keeping total element count
b <- reshape a                        -- type-driven: a :: Tensor '[2,3] 'F32 → b :: Tensor '[6] 'F32

-- Transpose: permute dimensions
c <- transpose a [1, 0]               -- [m,n] → [n,m]

-- Slice: extract a sub-array
d <- slice a [(0, 2), (1, 3)]        -- a[0:2, 1:3]

-- Pad: add padding around the edges
e <- pad a 0 [(1, 1), (0, 0)]        -- pad 1 on each side of dim 0

-- Concatenate: join tensors along a dimension
f <- concatenate @0 [a, b]            -- concat along dimension 0

-- Split: divide a tensor into N equal parts
[g, h] <- split @0 2 a               -- split dim 0 into 2 pieces

-- Stack: join along a NEW dimension
i <- stack @0 [a, b]                 -- adds a new dimension 0
```

### 4.6 Broadcasting

```haskell
-- Broadcast a scalar to a tensor shape
b <- broadcast scalar [2, 3]

-- Broadcast with explicit dimension mapping
c <- broadcastInDim a [0, 2] [4, 5, 6]   -- a has shape [4,6]; map dim 0→0, dim 1→2
```

Broadcasting follows NumPy/XLA semantics. The `broadcastInDim` op is the primitive; `broadcast` is a convenience wrapper.

---

## 5. Building and Executing

### 5.1 The `buildModule` Family

`buildModule` is the easiest way to create a compiled function:

```haskell
-- 1 input, 1 output
buildModule @1 @1 "f" $ \x -> ...

-- 2 inputs, 1 output
buildModule @2 @1 "f" $ \x y -> ...

-- 2 inputs, 2 outputs
buildModule @2 @2 "f" $ \x y -> returnTuple2 a b
```

For more than 2 outputs, use `buildModuleT` with the `Tuple` GADT:

```haskell
buildModuleT @3 @( '[s1,s2,s3], '[d1,d2,d3] ) "f" $ \x y z -> do
    return (t1 ::: t2 ::: t3 ::: TNil)
```

### 5.2 Raw Builder (Lower Level)

If you need full control, use the `Builder` monad directly:

```haskell
import HHLO.IR.Builder
import HHLO.IR.AST

myModule :: Module
myModule = moduleFromBuilder @'[3] @'F32 "main"
    [ FuncArg "x" (TensorType [3] F32) ]
    $ do
        x <- arg @'[3] @'F32
        y <- add x x
        return y
```

The `Builder` monad is a state monad that accumulates `Operation` values. `arg` declares a function argument. `emitOp` adds an operation. Most users never need this level, but it's there when you want to generate custom MLIR.

### 5.3 Compilation and Execution

```haskell
withCPU $ \sess -> do
    let model = buildModule @1 @1 "square" $ \x -> multiply x x
    compiled <- compile sess model

    input <- hostFromList @'[3] @'F32 [1, 2, 3]
    [output] <- run sess compiled [input]

    print (hostToList output)   -- [1.0, 4.0, 9.0]
```

`run` is synchronous. For async execution:

```haskell
bufs <- executeAsync api exec [inputBuf]
ready <- bufferReady api (head bufs)
-- ... do other work ...
awaitBuffers api bufs
results <- mapM (fromDeviceF32 api) bufs
```

---

## 6. Neural Network Primitives

HHLO provides common NN building blocks as typed combinators.

### 6.1 Convolution

```haskell
-- NHWC conv2d: input [N,H,W,C], kernel [kH,kW,C_in,C_out]
output <- conv2d input kernel

-- With explicit stride and padding
output <- conv2dWithPadding @1 @28 @28 @1 @3 @3 [2,2] [(1,1),(1,1)] input kernel
```

### 6.2 Pooling

```haskell
-- Max pool: [N,H,W,C] → [N,H',W',C]
pooled <- maxPool @1 @28 @28 @1 @2 @2 [2,2] [2,2] [(0,0),(0,0)] input

-- Average pool
pooled <- avgPool @1 @28 @28 @1 @2 @2 [2,2] [2,2] [(0,0),(0,0)] input

-- Global average pool: [N,H,W,C] → [N,1,1,C]
gap <- globalAvgPool input
```

### 6.3 Normalization

```haskell
-- Batch norm inference (decomposed into basic ops)
bn <- batchNormInference input scale offset mean variance epsilon

-- Layer norm
ln <- layerNorm input scale offset epsilon
```

### 6.4 Activation Helpers

```haskell
y <- relu x
y <- leakyRelu alpha x
y <- gelu x
y <- swish x
```

### 6.5 Putting It Together: A Mini ConvNet

```haskell
convBlock :: Tensor '[1,28,28,1] 'F32 -> Builder (Tensor '[1,14,14,32] 'F32)
convBlock input = do
    k1 <- constant @'[3,3,1,32]  @'F32 0.1
    c1 <- conv2d input k1
    r1 <- relu c1
    p1 <- maxPool @1 @28 @28 @32 @2 @2 [2,2] [2,2] [(0,0),(0,0)] r1
    return p1
```

---

## 7. Automatic Differentiation

This is where HHLO shines. Autograd is not a black-box C++ backend — it's a Haskell library that transforms StableHLO graphs.

### 7.1 Single-Parameter Gradients

```haskell
import HHLO.Autograd

-- f(x) = sum(x²)  =>  grad f(x) = 2x
gradMod :: Module
gradMod = gradModule @'[3] @'F32 $ \x -> do
    sq <- multiply x x
    sumAll sq
```

`gradModule` takes a scalar-valued function and returns a module that computes its gradient w.r.t. the input.

You can also use `grad` inside a larger builder:

```haskell
buildModule @1 @2 "loss_and_grad" $ \x -> do
    let loss = sumAll =<< multiply x x
    g <- grad (\y -> sumAll (multiply y y)) x
    returnTuple2 loss g
```

### 7.2 Multi-Parameter Gradients

For functions of multiple variables, use `grad2` and `grad3`:

```haskell
-- g(x,y) = sum(x * y)
-- grad_x = y, grad_y = x
(gradX, gradY) <- grad2 (\x y -> sumAll =<< multiply x y) xVal yVal
```

Similarly, `gradModule2` and `gradModule3` produce standalone modules:

```haskell
-- Module with 2 inputs, 2 outputs (gradients)
gradMod2 :: Module
gradMod2 = gradModule2 @'[2] @'F32 @'[2] @'F32 $
    \x y -> sumAll =<< multiply x y
```

### 7.3 Structured Parameters with ParamTree

Real models have dozens of weight tensors. Manually passing each one to `gradModule` is impractical. `ParamTree` solves this.

```haskell
{-# LANGUAGE DeriveGeneric #-}
import GHC.Generics (Generic)
import HHLO.Autograd

data MLPParams = MLPParams
    { w1 :: Tensor '[2,2] 'F32
    , b1 :: Tensor '[2]   'F32
    , w2 :: Tensor '[1,2] 'F32
    , b2 :: Tensor '[1]   'F32
    } deriving (Generic)

instance ParamTree MLPParams

forward :: MLPParams -> Tensor '[2] 'F32 -> Builder (Tensor '[1] 'F32)
forward p x = do
    h <- relu =<< add (matmul x (w1 p)) (b1 p)
    add (matmul h (w2 p)) (b2 p)

loss :: MLPParams -> Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
loss p x = do
    y <- forward p x
    target <- constant @'[1] @'F32 5.0
    diff <- sub y target
    sumAll =<< multiply diff diff

-- gradWithParams hides all packing/unpacking
trainStep :: MLPParams -> Tensor '[2] 'F32 -> Builder MLPParams
trainStep params x = gradWithParams loss params x
```

`ParamTree` uses `GHC.Generics` to derive the pack/unpack isomorphism automatically. Under the hood, it emits `slice`, `reshape`, and `concatenate` ops — all zero-copy views in XLA. There is **zero runtime overhead**.

### 7.4 Vector-Jacobian Products

For non-scalar outputs, use `vjp`:

```haskell
-- y = W @ x, where W is [2,3] and x is [3]
-- vjp f x seed = (Df(x))ᵀ · seed
vjpMod :: Module
vjpMod = vjpModule @'[3] @'[2] @'F32 $
    \x -> do w <- constant @'[2,3] @'F32 1.0; matmul w x
```

### 7.5 Supported Gradient Ops

VJP rules exist for: `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `log`, `sqrt`, `power`, `sine`, `cosine`, `tanh`, `abs`, `maximum`, `minimum`, `reshape`, `transpose`, `broadcast_in_dim`, `reduce` (sum), `dot`, `select`, `slice`, `pad`, `concatenate`, `convert`, `convolution`, `reduce_window` (max/avg pool), and more.

Ops without rules (e.g. `compare`, `floor`, `ceil`, `sort`) safely return zero gradients.

---

## 8. Control Flow

HHLO supports loops and conditionals via StableHLO regions.

### 8.1 While Loops

```haskell
-- whileLoop: condition and body are Builder actions
(result, finalSum) <- whileLoop2
    (0 :: Tensor '[1] 'I64, 0 :: Tensor '[] 'F32)
    (\c s -> do lt <- compare c limit "LT"; return lt)
    (\c s -> do
        cNext <- add c one
        sNext <- add s cNext
        returnTuple2 cNext sNext)
```

`whileLoop2` carries two typed values through the loop. The condition returns a `Tensor '[] 'Bool`. The body returns the next values.

Variants `whileLoop3` through `whileLoop8` support up to 8 loop-carried values.

### 8.2 Conditionals

```haskell
result <- conditional2
    predicate
    (\trueVal  -> do ... return something)
    (\falseVal -> do ... return something)
```

The true and false branches must return the same shape and dtype.

### 8.3 Random Number Generation

```haskell
-- Uniform [0, 1)
uniform <- rngUniform (constant @'[] @'F32 0.0) (constant @'[] @'F32 1.0)

-- Standard normal
normal <- rngNormal

-- Threefry bit generator (stateful)
(newState, bits) <- rngBitGenerator state
```

---

## 9. Multi-Device Execution

### 9.1 GPU Execution

```haskell
withGPU $ \sess -> do
    let model = buildModule @1 @1 "matmul" $ \x -> do
            w <- constant @'[4096,4096] @'F32 0.01
            matmul x w

    compiled <- compile sess model
    input <- hostFromList @'[4096,4096] @'F32 (replicate (4096*4096) 1.0)
    [output] <- run sess compiled [input]
    print (head (hostToList output))
```

The same code, just swap `withCPU` for `withGPU`. The CUDA plugin is auto-downloaded by `pjrt_script.sh` if `nvidia-smi` is present.

### 9.2 Multi-GPU Inference

Run the same compiled model concurrently across multiple GPUs:

```haskell
import HHLO.Runtime.Device (addressableDevices)
import HHLO.Runtime.Compile (compileWithOptions, defaultCompileOptions)
import HHLO.Runtime.Execute (executeReplicas)

multiGpuInfer :: IO ()
multiGpuInfer = withGPU $ \api client -> do
    devices <- addressableDevices api client
    let numDevs = length devices

    exec <- compileWithOptions api client mlirText
        (defaultCompileOptions { optNumReplicas = numDevs })

    -- Prepare one input buffer per GPU
    inputs <- mapM (\_ -> toDeviceF32 api client vec [4096,4096]) [1..numDevs]

    -- Execute concurrently
    outputs <- executeReplicas api exec
        [ (dev, [buf]) | (dev, buf) <- zip devices inputs ]

    results <- mapM (fromDeviceF32 api) outputs
    print (map (head . hostToList) results)
```

`executeReplicas` distributes independent forward passes across all available GPUs. This is **inference scaling**, not data-parallel training (which would require gradient synchronization).

### 9.3 Async Execution

For non-blocking execution:

```haskell
bufs <- executeAsync api exec [inputBuf]
-- Do other work...
awaitBuffers api bufs
results <- mapM (fromDeviceF32 api) bufs
```

`bufferReady` polls individual buffers for completion without blocking.

---

## 10. How It Works

Understanding the architecture helps you debug and extend HHLO.

### 10.1 The Five Layers

```
┌─────────────────────────────────────┐
│  Session (HHLO.Session)             │  One-liners: withCPU, compile, run
├─────────────────────────────────────┤
│  Autograd (HHLO.Autograd)           │  grad, vjp, ParamTree — reverse-mode AD
├─────────────────────────────────────┤
│  EDSL (HHLO.EDSL.Ops)               │  Type-safe frontend
├─────────────────────────────────────┤
│  IR Builder (HHLO.IR.Builder)       │  Stateful Builder monad
├─────────────────────────────────────┤
│  Pretty Printer (HHLO.IR.Pretty)    │  Emits StableHLO MLIR text
├─────────────────────────────────────┤
│  PJRT Runtime (HHLO.Runtime.*)      │  Compile → Execute on CPU/GPU
└─────────────────────────────────────┘
```

### 10.2 From Haskell to Executed Kernel

Here's the full pipeline when you call `run`:

1. **EDSL** — Your Haskell function `\x -> multiply x x` runs in the `Builder` monad, emitting `Operation` values into a list.
2. **Trace capture** (autograd only) — `gradModule` captures the forward trace as a list of ops, then runs the backward pass in reverse.
3. **Pretty printing** — `render` converts the `Module` AST to StableHLO MLIR text.
4. **PJRT compile** — `PJRT_Client_Compile` parses the MLIR, runs XLA optimization, and generates machine code.
5. **Execute** — `PJRT_Executable_Execute` launches the kernel on the target device.
6. **D2H transfer** — `fromDeviceF32` copies results back to host memory.

### 10.3 Autograd Internals

Autograd works via **source-to-source transformation** on the `Builder` monad:

1. **Forward trace** — `runBuilderWithTrace` records every emitted operation.
2. **Backward traversal** — The trace is reversed. For each forward op, its VJP rule emits gradient ops that propagate cotangents backward.
3. **Cotangent map** — A `Map ValueId BTensor` accumulates gradients for each value. If a value is used multiple times, its cotangents are added.
4. **Extraction** — `gradModule` extracts the gradient for the input argument(s).

The VJP rules live in `HHLO.Autograd.Rules`. Adding a new op's gradient means implementing one function:

```haskell
vjpMyOp :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpMyOp op resultBars cmap = do
    -- Emit backward ops...
    accumulate cmap operandVid gradient
```

### 10.4 PJRT Compatibility

The PJRT CPU plugin at `deps/pjrt/libpjrt_cpu.so` (StableHLO v1.16.0) has a few parser quirks that HHLO works around:

- `stablehlo.reduce` regions must use the generic form with explicit block args.
- `stablehlo.reverse` needs a custom pretty syntax (not the generic quoted form).
- `stablehlo.convolution` requires explicit `batch_group_count` and `feature_group_count`.
- Multi-result `func.func` is rejected (tuples are print-only on this parser version).

These are handled transparently by the pretty printer. Newer PJRT plugins or GPU backends accept the full StableHLO spec.

---

## 11. Appendix: Quick Reference

### Common Type Signatures

```haskell
-- EDSL ops
add         :: Tensor s d -> Tensor s d -> Builder (Tensor s d)
matmul      :: Tensor '[m,n] d -> Tensor '[n,p] d -> Builder (Tensor '[m,p] d)
conv2d      :: Tensor '[n,h,w,c] 'F32 -> Tensor '[kh,kw,c,o] 'F32 -> Builder (Tensor '[n,h',w',o] 'F32)
sumAll      :: Tensor s d -> Builder (Tensor '[] d)
constant    :: (KnownShape s, KnownDType d) => Double -> Builder (Tensor s d)

-- Autograd
gradModule  :: (KnownShape s, KnownDType d) => (Tensor s d -> Builder (Tensor '[] d)) -> Module
gradModule2 :: (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2) => (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor '[] d1)) -> Module
grad        :: (KnownShape s, KnownDType d) => (Tensor s d -> Builder (Tensor '[] d)) -> Tensor s d -> Builder (Tensor s d)
grad2       :: ... => (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor '[] d1)) -> Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1, Tensor s2 d2)
gradWithParams :: (ParamTree p, KnownShape s, KnownDType d) => (p -> Tensor s d -> Builder (Tensor '[] d)) -> p -> Tensor s d -> Builder p

-- Session
withCPU     :: (Session -> IO a) -> IO a
withGPU     :: (Session -> IO a) -> IO a
compile     :: Session -> Module -> IO CompiledModel
run         :: Session -> CompiledModel -> [HostTensor] -> IO [HostTensor]
```

### GHC Extensions You'll Need

```haskell
{-# LANGUAGE DataKinds        #-}  -- Promote data constructors to types
{-# LANGUAGE TypeApplications #-}  -- Pass types explicitly: @'[2,3]
{-# LANGUAGE TypeFamilies     #-}  -- Type-level functions (used internally)
{-# LANGUAGE ScopedTypeVariables #-}  -- Bring type variables into scope
{-# LANGUAGE DeriveGeneric    #-}  -- For ParamTree derivation
```

### Building and Running

```bash
# Download plugins
./pjrt_script.sh

# Build everything
cabal build all

# Run tests
cabal test                    # 190 CPU tests
cabal test --test-options="-t HHLO+GPU"  # + 6 GPU tests

# Run examples (requires --flag=examples)
cabal run example-autograd-basic --flag=examples
cabal run example-gpu-matmul-bench --flag=examples
```

---

*Document version: 1.0 | April 2026*

*For deeper architectural details, see `doc/implementation-design.md`. For the full API, explore the Haddocks or the source in `src/HHLO/`.*
