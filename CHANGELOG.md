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
