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


## Next -- 2026-04-27

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
* Test count: 169 CPU tests + 6 GPU integration tests.
