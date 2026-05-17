# Revision history for hhlo

## 0.1.0.0 -- 2026-04-22

* Initial release.
* Type-safe EDSL for StableHLO with 50+ ops.
* CPU execution via PJRT CPU plugin.
* GPU execution via PJRT CUDA plugin with device enumeration and selection.
* Multi-GPU concurrent inference scaling via `executeReplicas`.
* 115 CPU tests + 6 GPU integration tests.
* 29 executable examples including ResNet-18, AlexNet, Transformer, and UNet.

## 0.2.0.0 -- 2026-04-22

**BREAKING**: `Operation` AST changed from single-result to multi-result.
Any code using `opResult` / `opResultType` or pattern-matching on the
`Operation` constructor must update to `opResults` / `opResultTypes`.

* Multi-result `Operation` AST — `Operation` now supports `opResults :: [ValueId]`
  and `opResultTypes :: [TensorType]`, enabling ops with multiple outputs such as
  `stablehlo.rng_bit_generator`.
* Multi-value control flow — added `whileLoop2`, `conditional2`, `whileLoopN`,
  and `conditionalN` for carrying multiple typed tensors through loops and
  conditionals without manual packing.
* Random number generation — added `rngUniform`, `rngNormal`, and
  `rngBitGenerator` to the EDSL, wrapping `stablehlo.rng` and
  `stablehlo.rng_bit_generator`.
* PJRT CPU v1.16.0 parser compatibility fixes:
  * `stablehlo.compare` now emits generic form with enum attributes
    (`#stablehlo<comparison_direction LT>`) instead of custom form.
  * `stablehlo.rng` and `stablehlo.rng_bit_generator` emit generic form.
  * `func.return` for multi-result functions no longer wraps types in parentheses.
* New examples: `30-rng-uniform`, `31-rng-normal`, `32-rng-bit-generator`,
  `33-multi-value-loop`.
* Updated example `12-while` from print-only to fully executable.
* Test count: 124 CPU tests + 6 GPU integration tests.

## 0.3.0.0 -- 2026-04-25

**BREAKING**: `compare` and `lessThan` now return shape-preserving
`Tensor s 'Bool` instead of scalar `Tensor '[] 'Bool`. New exports
`sqrt`, `sin`, `cos`, `tan`, `floor`, `ceil` may conflict with Prelude.

* New primitive ops: `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `pow`, `log1p`, `floor`, `ceil`.
* New composite / convenience ops: `sigmoid`, `sumAll`, `pack2`, `pack3`, `slice1`.
* Fixed `compare` to return shape-preserving `Tensor s 'Bool` per StableHLO spec.
* New comparison wrappers: `equal`, `notEqual`, `greaterThan`, `lessThanOrEqual`, `greaterThanOrEqual`.
* Test count: 141 CPU tests + 6 GPU integration tests.

## 0.4.0.0 -- 2026-04-26

**BREAKING**: `HostType 'Bool` changed from `Bool` to `Word8` to match
PJRT's PRED buffer transfer semantics.

* **Convenience layer** — two new modules that eliminate boilerplate for
the common compile-and-run workflow:
  * `HHLO.ModuleBuilder` provides `buildModule @nIn @nOut`, a polymorphic
    entry point (via `TypeApplications`) that auto-generates `FuncArg`
    declarations and wires up `arg` calls. No more `natVal` or `FuncArg`
    boilerplate.
  * `HHLO.Session` provides `withCPU`, `withGPU`, `withGPUDevice`, `compile`,
    `run`, `runAsync`, and typed `HostTensor` host-device transfers.
    No more manual `render`, `toDeviceF32`, `fromDeviceF32`, or shape lists.
* `whileLoop3`–`whileLoop8` and `conditional3`–`conditional8` for carrying
  3–8 heterogeneous tensors through control flow.
* Boolean logic ops: `logicalAnd`, `logicalOr`, `logicalNot`.
* New dependency: `directory` (for plugin-path discovery in `withCPU`/`withGPU`).
* Test count: 155 CPU tests + 6 GPU integration tests.

## 0.5.0.0 -- 2026-04-27

* **Autograd** — reverse-mode automatic differentiation is now part of HHLO.
  New module `HHLO.Autograd` provides `grad` and `vjp` combinators that
  transform HHLO computation graphs into their gradients, producing new
  StableHLO modules that compile via PJRT. VJP rules cover ~25 ops including
  element-wise arithmetic, matmul, transpose, reshape, broadcast, reduce,
  slice, pad, concatenate, select, and more.
* New convenience ops:
  * `einsum` — Einstein summation via subscript strings (e.g. `"ij,jk->ik"`).
    Parses labels, computes batch/contracting dims, and emits the correct
    `stablehlo.dot_general` + optional `stablehlo.transpose`.
  * `split` — split a tensor into N equal parts along a dimension.
  * `stack` — stack N tensors along a new axis.
  * `productAll`, `productDim` — product reductions (mirrors `sumAll`/`reduceSumDim`).
  * `topK` — return top-K values along a dimension via `sort` + `slice`.
* Bug fix: `stablehlo.sort` now wraps its region in parentheses for PJRT
  v1.16.0 parser compatibility.
* Test count: 181 CPU tests + 6 GPU integration tests.

## 0.6.0.0 -- 2026-04-28
* **Convolution & pooling VJP rules** — autograd now supports backprop through
  `conv2d`, `transposeConvolution`, `maxPool`, and `avgPool`.
  * `vjpConvolution` / `vjpTransposeConvolution` emit backward input via
    flipped-kernel transposed conv and skip backward-kernel computation when
    the kernel is a constant (the common `gradModule` case).
  * `vjpReduceWindow` supports both sum-based (avgPool) and select-mask-based
    (maxPool) backward passes.
  * New primitive emitters: `bconvolution`, `breverse`.
  * PJRT parser compatibility: `stablehlo.reverse` custom pretty-printer and
    `batch_group_count` / `feature_group_count` attributes on backward convs.
  
## 0.7.0.0 -- 2026-04-28

* **Multi-parameter gradients** — `gradModule` is no longer limited to a single
  input. New combinators `gradModule2`, `gradModule3`, `grad2`, `grad3`
  differentiate w.r.t. multiple tensors natively.
* **ParamTree** — generic pack/unpack for structured parameter records.
  Derive via `GHC.Generics` and use `gradWithParams` to train models with
  dozens of weight tensors without manual offset math.
  ```haskell
  data MLPParams = MLPParams { w :: Tensor '[2,2] 'F32, b :: Tensor '[2] 'F32 }
      deriving (Generic)
  instance ParamTree MLPParams
  trainStep params x = gradWithParams loss params x
  ```

* New E2E autograd tests: `grad conv2d`, `grad maxPool`, `grad avgPool`,
  `grad2 multiply`, `gradWithParams`.
* New unit tests: `vjpConvolution`, `vjpTransposeConvolution`, `vjpReduceWindow`,
  `gradModule2`.
* Bug fix: `vjpSlice` padding value is now a 0D scalar (required by
  `stablehlo.pad`).
  
* **Comprehensive tutorial** — new document `doc/tutorial.md` (720 lines)
  providing a complete guided tour from `add` two scalars to multi-GPU
  distributed inference. Covers: shapes-as-types, the full EDSL, NN primitives,
  autograd (`grad`/`grad2`/`grad3`/ParamTree), control flow, async execution,
  and a deep-dive into the architecture and PJRT pipeline.
* Test count: 190 CPU tests + 6 GPU integration tests.

## 0.8.0.0 -- 2026-04-29

* **Nested ParamTree** — `ParamTree` now supports arbitrarily nested records
  via an overlapping `GParamTree (K1 R a)` instance. Fields can be other
  `ParamTree` records, not just bare `Tensor`s.
  ```haskell
  data LayerParams = LayerParams { w :: Tensor '[2] 'F32, b :: Tensor '[2] 'F32 }
      deriving (Generic)
  instance ParamTree LayerParams

  data ModelParams = ModelParams { layer1 :: LayerParams, layer2 :: LayerParams }
      deriving (Generic)
  instance ParamTree ModelParams
  ```
* New E2E autograd test: `gradWithParams nested`.
* Massive GPU test expansion — from 6 to 82 GPU integration tests.
  * New shared GPU test harness (`Test.Runtime.GPUResource`) using `tasty`
    `withResource` for a single PJRT client shared across all GPU tests.
  * GPU counterparts for nearly all CPU EndToEnd test categories:
    Arithmetic (15), Shape (8), Matmul (6), NN (7), Reductions (5),
    DataMovement (15), MultiValue (6), Autograd (10), Session (4).
  * New typed GPU helpers: `toDeviceF32On`, `toDevicePredOn`, `toDeviceS64On`.
  * Total: 191 CPU tests + 82 GPU tests = 273 tests.
* New `sessionFrom` constructor in `HHLO.Session` — create a `Session` from an
  existing PJRT API/client/device without loading a new plugin.
* Fixed `CUDA_ERROR_OUT_OF_MEMORY` warnings during GPU tests by converting
  `SessionGPU` tests to reuse the shared PJRT client (was creating 4 separate
  clients via `withGPU`, each contending for the same GPU memory).
* Moved `getPluginPath` from `HHLO.Session` to `HHLO.Runtime.PJRT.Plugin`.
  `withPJRTCPU` and `withPJRTGPU` now resolve plugin paths via the
  `HHLO_PJRT_CPU_PLUGIN` / `HHLO_PJRT_GPU_PLUGIN` environment variables
  (falling back to `deps/pjrt/`), so downstream libraries no longer need to
  reimplement plugin discovery.

## 0.9.0.0 -- 2026-04-20

* **Fixed-length configuration vectors** — rank-polymorphic EDSL ops now use
  `vector-sized` to tie config vector lengths to tensor ranks at compile time.
  This eliminates the class of bugs where wrong-length config silently produces
  invalid StableHLO.
  * Phase 1 (2D NN primitives): `conv2dWithPadding`, `maxPool`, `avgPool`,
    `transposeConvolution` accept `V2 Int64` / `P2`.
  * Phase 2+ (rank-polymorphic ops): `transpose`, `slice`, `pad`,
    `dynamicSlice`, `reduceWindow`, and `dotGeneral` now accept
    `Vector (Length s) Int64` or separate `Vector n Int64` type parameters
    instead of raw `[Int64]`.
  ```haskell
  -- BEFORE (could silently miscompile)
  transpose [1, 0] x
  slice x [1] [3] [1]
  dotGeneral [] [] [1] [0] x y

  -- AFTER (type-safe)
  transpose (v2 1 0) x
  slice x (v1 1) (v1 3) (v1 1)
  dotGeneral VS.empty VS.empty (v1 1) (v1 0) x y
  ```
  New exports in `HHLO.Core.Types`: `Length` type family, `V`, `V1`, `V2`,
  `V3`, `V4`, `Padding`, `P2`, plus smart constructors `v1`, `v2`, `v3`, `v4`,
  `p2`.
* New dependency: `vector-sized >= 1.5 && < 1.6`.
* Fix `transposeConvolution` lhs_dilation bug — passing a 2-element spatial
  dilation list no longer drops the second element.
  
## 0.10.0.0 -- 2026-05-02
* Fix `vjpConcatenate` slice offset bug — `splitAndAccumulate` now uses the
  cumulative `offset` in `start_indices` for all operands after the first.
  Previously only `limit_indices` used the offset, causing PJRT to reject
  gradient modules for any `concatenate2` with 2+ operands.
* Fix three autograd padding bugs:
  1. `transposeConvBackwardInput` — now applies `reversePad` so backward
     regular conv uses reversed padding instead of forward padding directly.
  2. `convBackwardInput` with `stride > 1` — now computes correct
     `targetPadTotal = input - (bar-1)*stride + kernel - 2` and adjusts
     reverse-pad to account for XLA floor division, restoring the original
     input spatial size in the transpose-conv backward pass.
  3. `vjpSlice` with `stride > 1` — `high'` padding now uses
     `n = ceil((limit-start)/stride)` instead of raw `(limit-start)`,
     preventing over-padded gradients.
* `gather` and `scatter` kept as `[Int64]` for now. Their config vector lengths
  depend on complex relationships between operand / indices / result ranks, so
  a clean type-safe design requires a separate future phase.
* New E2E tests exercising the vector-sized configs:
  * `grad concatenate2`, `grad concatenate3` (regression tests for the concat bug)
  * `slice 2D`, `pad 2D symmetric`
  * `dotGeneral batched`
  * `transpose 3D`
  * GPU counterparts for all of the above.
* New regression tests for the padding bugs:
  * `grad conv2d stride` — strided conv `[1,4,4,1]` with stride=2
  * `grad transposeConvolution asymmetric pad` — transpose conv with
    `p2 (1,0) (1,0)` and `v2 2 2`
  * `grad pad interior` — pad with interior=1 on `[2]`
  * `transposeConvolution forward` — basic transpose conv smoke test
  * `conv2dWithPadding forward` — strided conv with explicit padding
* **Generic custom-call infrastructure** — first-class support for
  `stablehlo.custom_call`, enabling external packages to register their own
  XLA custom-call kernels (CUDA, CPU, or otherwise) without modifying HHLO.
  * `HHLO.IR.Builder.emitCustomCall` — emit `stablehlo.custom_call` with
    standard attributes (`call_target_name`, `has_side_effect`,
    `backend_config`, `api_version`).
  * `HHLO.EDSL.Ops.customCall1`, `customCall2`, `customCallRaw` — typed
    frontend wrappers. `customCallRaw` is the escape hatch for heterogeneous
    input types and arbitrary result counts.
  * `HHLO.Runtime.CustomCall.loadCustomCallLibrary` — `dlopen` wrapper with
    `RTLD_GLOBAL`, required for XLA's internal `dlsym` resolution.
  * `HHLO.IR.Pretty` — special case for `stablehlo.custom_call` emits the
    `@symbol` prefix before operands (e.g.
    `stablehlo.custom_call @foo(%arg0) {...} : ...`).
  * `examples/CustomCallPlugin.hs` + `examples/cbits/vector_add.cu` —
    minimal working example showing the full plugin contract.
* Test count: 205 CPU tests + 82 GPU tests = 287 total.
* **Bug fix**: Full GPU test suite (`HHLO_TEST_GPU=1 cabal test`) no longer
  crashes with mutex corruption from the PJRT CUDA plugin. Root cause was the
  `EndToEnd.Dynamic` "dynamic add on GPU" test creating a separate PJRT client
  via `withGPU` in the CPU tree; when that client was destroyed, buffer and
  executable finalizers could race with plugin teardown, corrupting internal
  mutex state. Fixed by moving the GPU dynamic test to the shared `GPUResource`
  tree (new `Test.Runtime.EndToEndDynamicGPU` module). Also added missing
  `Test.Runtime.EndToEndDynamic` entry to `.cabal` `other-modules`.
* Test count after fix: 216 CPU tests + 95 GPU tests = 311 total.

## 0.11.0.0 -- 2026-05-17

* **Dynamic shape support** — tensors whose dimensions are only known at
  runtime can now be constructed, compiled, and executed end-to-end.
  * `HHLO.EDSL.Dynamic` provides dynamic-shape variants of core ops
    (`dynamicAdd`, `dynamicMatmul`, `dynamicConv2d`, etc.) using the new
    `AnyTensor` type.
  * `HHLO.Session.Dynamic` provides `runDynamic` and `runDynamicAsync`
    for transferring and executing `DynamicHostTensor` values.
  * `HHLO.IR.AST.TensorType` now uses `[Maybe Integer]` for dimensions
    (`Nothing` renders as `?` in MLIR), enabling static/dynamic mixed shapes.
  * New example: `37-dynamic-shapes.hs`.
  * New tests: `Test.Runtime.EndToEndDynamic` and
    `Test.Runtime.EndToEndDynamicGPU`.

* **Device assignment compile option** — `CompileOptions` gains
  `optDeviceAssignment`, and the `Session` API now routes buffer transfers
  and execution to the selected device rather than defaulting to the first
  device. Fixes multi-GPU session behaviour.
  * C shim (`cbits/pjrt_shim.c`) extended to pass device assignment through
    to `PJRT_Client_Compile`.

* **ForeignPtr lifetime hardening** — critical FFI safety fix for premature
  GC finalization.
  * Added `withBufferPtr`, `withExecPtr`, and `withBufferPtrs` helpers to
    `HHLO.Runtime.PJRT.Types`.
  * `Execute`, `Buffer`, and `Async` modules hardened so that PJRT buffers
    and executables cannot be finalized while still in use by C calls.

* **Global API registry** — buffer finalizers are now safe even after the
  PJRT session closes.
  * New module `HHLO.Runtime.PJRT.Registry` maintains a global registry of
    active PJRT API pointers.
  * Prevents use-after-free crashes when GC runs after plugin teardown.
  * New regression tests in `Test.Runtime.Buffer` and `Test.Runtime.BufferGPU`.

* **Structured attribute generation** — all attribute generation across
  the EDSL, Pretty printer, Autograd, and IR Builder now uses a typed,
  structured approach instead of ad-hoc attribute lists.
  * New module: `HHLO.ShapeCheck` (~950 lines) for centralized compile-time
    shape validation logic.

* **Matmul unified on `dot_general`** — `matmul` now emits canonical
  `dot_general` attributes. Added `vjpDotGeneral` supporting arbitrary-rank
  batched matrix multiplication, with new E2E tests for batched configs.

* **Zero-argument module execution** — `ToDeviceInputs` and
  `FromDeviceOutputs` are now exported from `HHLO.Session`, and `()`
  instances allow compiling and executing modules with no inputs or no
  outputs (e.g. constant generators).

* GPU test harness improvements:
  * SessionGPU regression tests refactored to reuse the shared GPU client,
    eliminating BFC allocator retry noise.
  * Dynamic-shape GPU tests moved into the shared `GPUResource` tree.

* Test count: 218 CPU tests; GPU suite expanded with additional session,
  buffer, and dynamic-shape coverage.
