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
* New primitive ops: `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `pow`, `log1p`, `floor`, `ceil`.
* New composite / convenience ops: `sigmoid`, `sumAll`, `pack2`, `pack3`, `slice1`.
* Fixed `compare` to return shape-preserving `Tensor s 'Bool` per StableHLO spec.
* New comparison wrappers: `equal`, `notEqual`, `greaterThan`, `lessThanOrEqual`, `greaterThanOrEqual`.
* Test count: 141 CPU tests + 6 GPU integration tests.
