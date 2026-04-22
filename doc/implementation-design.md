# HHLO Implementation Design

> **The Architecture of a Haskell→StableHLO Compiler**
>
> *A reference document describing how HHLO transforms Haskell expressions into executed machine code via StableHLO and PJRT.*

---

## 1. Executive Summary

HHLO is a layered compiler that translates type-safe Haskell expressions into hardware-agnostic StableHLO MLIR, then compiles and executes that MLIR via the PJRT plugin interface. The system has four layers:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — EDSL          (HHLO.EDSL.Ops)                    │
│  Type-safe frontend: tensors with phantom shape/dtype types │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — IR Builder    (HHLO.IR.Builder + AST)            │
│  Stateful monad for constructing MLIR operations            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Pretty Printer (HHLO.IR.Pretty)                  │
│  Emits StableHLO MLIR text with custom & generic forms      │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 — PJRT Runtime  (HHLO.Runtime.*)                   │
│  Compile MLIR → execute on CPU/GPU/TPU via plugins          │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Text emission, not MLIR C API.** We emit StableHLO MLIR as text and let PJRT parse it. This eliminates the LLVM/MLIR build dependency entirely.
- **Phantom types for shapes.** `Tensor '[2, 3] 'F32` carries shape and dtype in the type, enabling compile-time shape checking via type families.
- **Generic region form for nested ops.** `stablehlo.reduce`, `stablehlo.while`, `stablehlo.if`, etc. emit proper MLIR regions with basic blocks, ensuring compatibility with all PJRT parsers.
- **Thin C shim.** A ~300-line C file wraps the PJRT C API vtable into flat functions that Haskell FFI can call directly.

---

## 2. Design Principles

| Principle | How it manifests |
|-----------|-----------------|
| **Type safety first** | Every tensor carries its shape and dtype as phantom type parameters. Matmul, broadcast, and conv shapes are checked at compile time via type families. |
| **No heavy dependencies** | No LLVM, MLIR, or `mlir-hs` build required. Only GHC, a C compiler, and a prebuilt PJRT plugin. |
| **Text as the IR boundary** | MLIR text is the interchange format between Layer 3 (Pretty) and Layer 4 (Runtime). PJRT parses it internally. |
| **PJRT as the hardware abstraction** | CPU, GPU, and TPU are all accessed through the same PJRT C API. The Haskell code is backend-agnostic. |
| **Regions for nested ops** | Control flow and reductions use MLIR regions (`{ ^bb0(...) : ... }`) rather than ad-hoc shorthands, ensuring parser compatibility. |
| **ForeignPtr finalizers** | PJRT buffers and executables are managed by GHC's garbage collector via `ForeignPtr` finalizers. |

---

## 3. End-to-End Data Flow

```
Haskell program
      │
      ▼
┌─────────────────┐
│  EDSL Ops       │  add, matmul, conv2d, softmax, ...
│  (type-safe)    │  Phantom types: Tensor '[2,3] 'F32
└─────────────────┘
      │ Builder (Tensor s d) → Builder (Tensor s' d)
      ▼
┌─────────────────┐
│  IR Builder     │  Stateful monad accumulating:
│  (AST in mem)   │  - Operations (value ids, operands, attrs)
│                 │  - Regions (blocks with args + ops)
│                 │  - Next fresh value id
└─────────────────┘
      │ render :: Module → Text
      ▼
┌─────────────────┐
│  Pretty Printer │  Emits StableHLO MLIR text:
│  (MLIR text)    │  module { func.func @main(...) { ... } }
└─────────────────┘
      │ PJRT_Client_Compile
      ▼
┌─────────────────┐
│  PJRT Plugin    │  Parses MLIR → StableHLO → MHLO → HLO
│  (C API)        │  → XLA optimizations → LLVM IR → machine code
└─────────────────┘
      │ execute
      ▼
┌─────────────────┐
│  Hardware       │  CPU (now) / GPU / TPU (future)
└─────────────────┘
```

---

## 4. Layer 1 — EDSL (`HHLO.EDSL.Ops`)

### 4.1 The Tensor Type

```haskell
newtype Tensor (s :: Shape) (d :: DType) = Tensor
    { tensorValue :: ValueId   -- internal MLIR value identifier
    }
```

`s` is a type-level list of naturals (`'[2, 3]`). `d` is a promoted `DType` (`'F32`, `'I64`, `'Bool`). The actual data lives on the device; `Tensor` is just a typed reference to an MLIR value.

### 4.2 DType and Host Type Mapping

```haskell
data DType = F32 | F64 | I8 | I16 | I32 | I64
           | UI8 | UI16 | UI32 | UI64
           | Bool

type family HostType (d :: DType) :: Type where
    HostType 'F32  = Float
    HostType 'F64  = Double
    HostType 'I32  = Int32
    HostType 'I64  = Int64
    HostType 'Bool = Bool
```

`HostType` bridges between MLIR dtypes and Haskell types for buffer transfer.

### 4.3 Shape Inference via Type Families

Shape inference happens entirely at compile time:

```haskell
type family MatMulShape (s1 :: Shape) (s2 :: Shape) :: Shape where
    MatMulShape '[m, n] '[n, p] = '[m, p]
    MatMulShape '[n]    '[n, p] = '[p]

type family ReduceAllShape (s :: Shape) :: Shape where
    ReduceAllShape '[]       = '[]
    ReduceAllShape (_ ': xs) = ReduceAllShape xs
```

Type errors for invalid shapes are standard GHC errors:
```
Couldn't match type '20' with '30'
Expected: Tensor '[10, 30] F32
  Actual: Tensor '[10, 20] F32
```

### 4.4 Op Categories

The EDSL provides 50+ ops organized into categories:

| Category | Ops | Shape Constraint |
|----------|-----|-----------------|
| Element-wise unary | `relu`, `negate`, `abs'`, `exponential`, `logarithm`, `tanh`, `erf` | output shape = input shape |
| Element-wise binary | `add`, `sub`, `multiply`, `divide`, `maximum`, `minimum` | both inputs same shape |
| Matmul | `matmul`, `dotGeneral`, `linear`, `linearBatched` | `MatMulShape` type family |
| Convolution | `conv2d`, `conv2dWithPadding` | NHWC input, HWCF kernel |
| Reduction | `reduceSum`, `reduceSumDim`, `maxPool`, `avgPool`, `globalAvgPool` | dimension list reduces shape |
| Shape | `reshape`, `transpose`, `broadcastWithDims`, `concatenate`, `concatenate2`, `slice`, `pad`, `dynamicSlice` | explicit output shape |
| NN layers | `softmax1D`, `softmax2D`, `batchNormInference`, `layerNorm`, `gelu` | composite (built from primitives) |
| Control flow | `whileLoop`, `conditional`, `compare`, `sort`, `map` | region-based |
| Data movement | `gather`, `scatter`, `select`, `convert`, `iota` | attribute-heavy |

---

## 5. Layer 2 — IR Builder (`HHLO.IR.Builder` + `HHLO.IR.AST`)

### 5.1 The Builder Monad

`Builder` is a `State` monad that accumulates operations and tracks the next fresh value id:

```haskell
data BuildState = BuildState
    { bsNextId   :: !Int          -- next SSA value id (%0, %1, ...)
    , bsOps      :: ![Operation]  -- accumulated ops (in reverse)
    , bsArgCount :: !Int          -- next argument index (for %argN)
    }

newtype Builder a = Builder (State BuildState a)
```

### 5.2 Operation Emission

```haskell
emitOp :: Text -> [ValueId] -> [TensorType] -> [Attribute]
       -> TensorType -> Builder ValueId
```

Emits a single MLIR operation and returns its result value id. For example, `emitOp "stablehlo.add" [v1, v2] [t1, t2] [] tOut` produces:
```mlir
%3 = stablehlo.add %1, %2 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
```

### 5.3 Region-Bearing Operations

For nested ops (`while`, `if`, `reduce`, `sort`, `map`, `scatter`), we use `emitOpRegions`:

```haskell
emitOpRegions :: Text -> [ValueId] -> [TensorType] -> [Attribute]
              -> [Region] -> TensorType -> Builder ValueId
```

Regions contain `Block`s, which contain their own operations and argument lists. `runBlockBuilder` creates an isolated block builder that shares the global value id counter but resets the argument counter:

```haskell
runBlockBuilder :: [TensorType] -> Builder a -> Builder Block
```

This ensures block arguments get fresh names (`%arg0`, `%arg1` within the block) even when nested inside a function that already has arguments.

### 5.4 Tuple Support

Multi-result functions are supported via a `Tuple` GADT:

```haskell
data Tuple (ss :: [Shape]) (ds :: [DType]) where
    TNil  :: Tuple '[] '[]
    (:::) :: Tensor s d -> Tuple ss ds -> Tuple (s ': ss) (d ': ds)
```

`moduleFromBuilderT` builds a function that returns multiple results, which the pretty-printer emits as a `func.func` with multiple return types.

---

## 6. Layer 3 — Pretty Printer (`HHLO.IR.Pretty`)

The pretty-printer converts the in-memory AST to StableHLO MLIR text. It handles two output styles:

### 6.1 Custom Form (concise)

For common ops, we emit a custom MLIR syntax that matches what StableHLO documentation uses:

```mlir
%0 = stablehlo.add %arg0, %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>

%1 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b,0,1,f]x[0,1,i,o]->[b,0,1,f],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]]}
    : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x4x4x1xf32>

%2 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0] x [0],
    contracting_dims = [2] x [1]
    : (tensor<1x2x3xf32>, tensor<1x3x2xf32>) -> tensor<1x2x2xf32>
```

### 6.2 Generic Form (universal)

For ops that PJRT's parser handles better in generic form, or when we need regions:

```mlir
%0 = "stablehlo.reduce"(%arg0, %cst) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
}) {dimensions = array<i64: 0>} : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
```

**Why both forms?** The custom form is human-readable and matches the StableHLO spec. The generic form is required for region-bearing ops and for compatibility with PJRT parsers that don't implement every custom syntax variant.

### 6.3 Integer Constant Formatting

A critical detail: `dense<0.0>` is invalid for integer tensors in StableHLO. The pretty-printer formats values according to dtype:

| DType | Emits |
|-------|-------|
| `F32`, `F64` | `0.0`, `3.14` |
| `I64`, `I32`, `I16`, `I8` | `0`, `42` |
| `Bool` | `true`, `false` |

---

## 7. Layer 4 — PJRT Runtime (`HHLO.Runtime.*`)

### 7.1 The C Shim (`cbits/pjrt_shim.c`)

PJRT uses a vtable-style C API: every function is a function pointer inside a struct. Haskell's FFI cannot call vtable functions directly. The C shim extracts function pointers and exposes flat C functions:

```c
// PJRT API: api->PJRT_Client_Create(...)
// C shim:   hhlo_pjrt_client_create(api, ...)
```

The shim also handles:
- Plugin loading via `dlopen`
- API struct initialization
- Buffer type constant getters (16 dtype enum values)
- Event polling and awaiting
- Buffer metadata queries (dimensions, element type, on-device size)

### 7.2 FFI Layer (`HHLO.Runtime.PJRT.FFI`)

Haskell FFI imports the flat C functions:

```haskell
foreign import ccall "pjrt_shim.h hhlo_pjrt_load_plugin"
    c_pjrtLoadPlugin :: CString -> Ptr (Ptr PJRTApi) -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_create_client"
    c_pjrtCreateClient :: Ptr PJRTApi -> Ptr (Ptr PJRTClient) -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_client_compile"
    c_pjrtClientCompile :: Ptr PJRTApi -> Ptr PJRTClient
                        -> CString -> CInt
                        -> Ptr (Ptr PJRTLoadedExecutable)
                        -> IO (Ptr PJRTError)
```

### 7.3 Error Handling

PJRT errors are converted to Haskell exceptions:

```haskell
checkError :: Ptr PJRTApi -> IO (Ptr PJRTError) -> IO ()
checkError api action = do
    err <- action
    if err == nullPtr
        then return ()
        else withErrorMessage api err >>= throwIO . PJRTException
```

### 7.4 Compilation Pipeline

```haskell
compile :: PJRTApi -> PJRTClient -> Text -> IO PJRTLoadedExecutable
```

1. Render `Module` to `Text`
2. Encode as UTF-8 `CString`
3. Call `hhlo_pjrt_client_compile`
4. Wrap the result in a `ForeignPtr` with a finalizer that calls `PJRT_LoadedExecutable_Destroy`

### 7.5 Execution

**Synchronous:**
```haskell
execute :: PJRTApi -> PJRTLoadedExecutable -> [PJRTBuffer] -> IO [PJRTBuffer]
```

Queries the executable for its output count via `PJRT_Executable_NumOutputs`, allocates result buffers, calls `PJRT_Executable_Execute`, and wraps outputs in `ForeignPtr` finalizers.

**Asynchronous:**
```haskell
executeAsync :: PJRTApi -> PJRTLoadedExecutable -> [PJRTBuffer]
             -> IO [PJRTBuffer]
```

Returns buffer handles immediately. Use `bufferReady` to poll or `awaitBuffers` to block until completion.

### 7.6 Buffer Management

```haskell
toDeviceF32 :: PJRTApi -> PJRTClient -> Vector Float -> [Int64] -> IO PJRTBuffer
fromDeviceF32 :: PJRTApi -> PJRTBuffer -> Int -> IO (Vector Float)
```

Buffers are `ForeignPtr`-wrapped PJRT handles. When the Haskell value is garbage-collected, the finalizer calls `PJRT_Buffer_Destroy`.

Buffer metadata queries:
```haskell
bufferDimensions    :: PJRTApi -> PJRTBuffer -> IO [Int64]
bufferElementType   :: PJRTApi -> PJRTBuffer -> IO CInt
bufferOnDeviceSize  :: PJRTApi -> PJRTBuffer -> IO CSize
```

---

## 8. Type System Design

### 8.1 Static Shapes (Primary Mode)

The normal mode of operation. All dimensions are type-level `Nat`s:

```haskell
program :: Module
program = moduleFromBuilder @'[2, 2] @'F32 "main"
    [ FuncArg "a" (TensorType [2, 2] F32)
    , FuncArg "b" (TensorType [2, 2] F32)
    ]
    $ do
        a <- arg @'[2, 2] @'F32
        b <- arg @'[2, 2] @'F32
        c <- add a b
        return c
```

Shape mismatches are caught at compile time by the `MatMulShape`, `BroadcastResult`, etc. type families.

### 8.2 Dynamic Shape Escape Hatch

For shapes unknown at compile time (e.g., variable batch size), the user can:

1. Build the module at runtime with explicit shape values
2. Use `moduleFromBuilder` with type applications determined at runtime

Since `moduleFromBuilder` is a regular function (not Template Haskell), it can be called with any shape:

```haskell
buildForBatch :: Int -> Module
buildForBatch batchSize =
    -- Use a GADT or existential to hide the shape at the type level
    -- then unsafeCoerce to provide the KnownShape instance
    ...
```

This is an advanced pattern. Most users work with static shapes.

### 8.3 Type-Level Constraints

```haskell
-- Element-wise ops require identical shapes
add :: (s1 ~ s2, d1 ~ d2) => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)

-- Matmul requires compatible shapes via type family
matmul :: (KnownShape s1, KnownShape s2, KnownShape (MatMulShape s1 s2))
       => Tensor s1 F32 -> Tensor s2 F32 -> Builder (Tensor (MatMulShape s1 s2) F32)
```

---

## 9. Testing Architecture

The test suite is organized in three tiers, independent of the library layers:

| Tier | Strategy | What it validates | Speed |
|------|----------|-------------------|-------|
| **Golden** | Build AST → render → assert text contains expected MLIR | Pretty-printer correctness, op emission | ~1 ms |
| **E2E Numerical** | Full pipeline: EDSL → AST → Pretty → PJRT compile → execute → verify | End-to-end stack correctness | ~20 ms |
| **Integration** | Buffer round-trips, async events, error paths | Runtime FFI correctness | ~20 ms |

**Test modules:**
- `Test.EDSL.Ops` — Golden tests for every EDSL op
- `Test.IR.Pretty*` — Golden tests for pretty-printer special cases
- `Test.IR.Builder` — Builder state invariants
- `Test.Runtime.EndToEnd*` — Numerical verification per op category
- `Test.Runtime.Buffer` / `Async` / `Errors` — Runtime integration

See `doc/test-suite-documentation.md` for the complete test catalog.

---

## 10. Extension Points

The architecture is designed to accommodate these future directions without redesign:

### 10.1 GPU / Multi-Device Support

PJRT plugins for CUDA, ROCm, and TPU all implement the same C API. To target GPU:

1. Download `libpjrt_cuda.so` from `zml/pjrt-artifacts`
2. Load it instead of (or alongside) the CPU plugin
3. Pass device IDs to `execute` via `PJRT_ExecuteOptions`

No changes to Layers 1–3 are needed. Only Layer 4 needs device enumeration APIs.

### 10.2 Automatic Differentiation

Reverse-mode autodiff can be added as a source-to-source transformation on the `Builder` monad:

1. Record the computation graph during `Builder` execution
2. Traverse the graph backward, emitting gradient ops
3. Use `stablehlo.custom_call` for ops without native gradient definitions

This is analogous to JAX's `jax.grad` but operates on the AST level rather than tracing Python.

### 10.3 Template Haskell Staging

For compile-time MLIR generation (AOT compilation):

```haskell
compiled :: Tensor '[Batch, 784] 'F32 -> IO (Tensor '[Batch, 10] 'F32)
compiled = $(compileTH [|| myModel ||])
```

A TH splice would run the `Builder` action at compile time, render the MLIR, call PJRT compile, and embed the resulting executable bytecode into the binary. This removes first-call JIT latency entirely.

### 10.4 Model Serialization

PJRT supports `PJRT_Executable_Serialize` and `PJRT_Executable_Deserialize`. Adding these to the C shim would enable:

- Caching compiled programs to disk
- Distributing precompiled models without the source
- Faster startup (skip compilation on subsequent runs)

### 10.5 Higher-Level Layer Library

The EDSL currently exposes primitive ops. A higher-level module could provide:

```haskell
linear :: Int -> Int -> Tensor '[b, in] 'F32 -> Builder (Tensor '[b, out] 'F32)
conv2dLayer :: Int -> Int -> Int -> Tensor '[b,h,w,c] 'F32 -> Builder (Tensor '[b,h',w',c'] 'F32)
```

These are pure Haskell combinators built on top of the existing primitives.

---

## 11. Module Reference

| Module | Purpose | Key Types / Functions |
|--------|---------|----------------------|
| `HHLO.Core.Types` | DType, Shape, HostType | `DType`, `Tensor`, `HostType`, `KnownShape` |
| `HHLO.IR.AST` | MLIR AST | `Operation`, `Function`, `Module`, `Block`, `Region`, `Attribute` |
| `HHLO.IR.Builder` | Stateful builder | `Builder`, `emitOp`, `emitOpRegions`, `runBlockBuilder`, `arg`, `moduleFromBuilder` |
| `HHLO.IR.Pretty` | MLIR text emission | `Pretty`, `render`, `denseElements` |
| `HHLO.EDSL.Ops` | User-facing ops | `add`, `matmul`, `conv2d`, `softmax`, `whileLoop`, `conditional`, ... |
| `HHLO.Runtime.PJRT.FFI` | C FFI | `c_pjrtLoadPlugin`, `c_pjrtClientCompile`, `c_pjrtExecute` |
| `HHLO.Runtime.PJRT.Types` | Opaque handles | `PJRTApi`, `PJRTClient`, `PJRTBuffer`, `PJRTLoadedExecutable` |
| `HHLO.Runtime.PJRT.Error` | Error handling | `checkError`, `PJRTException` |
| `HHLO.Runtime.Compile` | Compilation | `compile` |
| `HHLO.Runtime.Execute` | Sync execution | `execute` |
| `HHLO.Runtime.Async` | Async execution | `executeAsync`, `bufferReady`, `awaitBuffers` |
| `HHLO.Runtime.Buffer` | Buffer transfer | `toDeviceF32`, `fromDeviceF32`, `bufferDimensions` |

---

*Document Version: 2.0 | April 2026*
