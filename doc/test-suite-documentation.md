# HHLO Test Suite — Comprehensive Documentation

**Date:** 2026-04-20  
**Test Count:** 115 tests across 13 modules  
**Framework:** `tasty` + `tasty-hunit`  
**Entry Point:** `test/Main.hs` → `test-suite hhlo-test` in `hhlo.cabal`

---

## 1. Overview

The HHLO test suite validates every layer of the stack:

| Tier | What | Files | PJRT Required? |
|------|------|-------|----------------|
| **Tier 1 — Golden** | Rendered MLIR text matches expected StableHLO | `Test.IR.Pretty*`, `Test.EDSL.Ops`, `Test.IR.Builder` | ❌ No |
| **Tier 2 — E2E Numerical** | Build → Compile → Execute → Verify on CPU | `Test.Runtime.EndToEnd*` | ✅ Yes |
| **Tier 3 — Runtime Integration** | Buffer metadata, async execution, error handling | `Test.Runtime.Buffer`, `Test.Runtime.Async`, `Test.Runtime.Errors` | ✅ Yes |

All E2E tests use the PJRT CPU plugin at `deps/pjrt/libpjrt_cpu.so` (downloaded via `./pjrt_script.sh`).

---

## 2. Shared Infrastructure

### `test/Test/Utils.hs`

This module provides the backbone for every E2E and integration test.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `withPJRTCPU` | `(PJRTApi → PJRTClient → IO a) → IO a` | Loads CPU plugin, creates client, runs action, cleans up |
| `e2eTestF32_1arg` | `String → Vector Float → (Tensor '[2,2] 'F32 → Builder (Tensor '[2,2] 'F32)) → Vector Float → TestTree` | Wraps a unary 2×2 F32 op in a full build/compile/execute/verify pipeline |
| `e2eTestF32_2arg` | `String → Vector Float → Vector Float → (Tensor '[2,2] 'F32 → Tensor '[2,2] 'F32 → Builder (Tensor '[2,2] 'F32)) → Vector Float → TestTree` | Same for binary 2×2 F32 ops |
| `moduleFromBuilder22` | `(Tensor '[2,2] 'F32 → Tensor '[2,2] 'F32 → Builder (Tensor '[2,2] 'F32)) → Module` | Convenience for 2-arg 2×2 modules |
| `moduleFromBuilder1_22` | `(Tensor '[2,2] 'F32 → Builder (Tensor '[2,2] 'F32)) → Module` | Convenience for 1-arg 2×2 modules |
| `goldenTest` | `String → Text → Text → TestTree` | Simple text-equality test |
| `assertThrowsPJRT` | `String → IO a → TestTree` | Asserts that an action throws `PJRTException` |

**Important:** Every E2E test that needs PJRT must be wrapped in `withPJRTCPU`. The plugin is loaded fresh per test case, so tests are isolated but not fast (~10–30 ms each).

---

## 3. Tier 1 — Golden Tests

Golden tests build a computation via the EDSL or raw `Builder`, render it to `Text`, and assert that the output contains expected MLIR substrings. They require no external dependencies and run in milliseconds.

### 3.1 `test/Test/IR/Pretty.hs`

Tests the core pretty-printer for `TensorType`, `Operation`, `Function`, and `Module`.

| Test | What it verifies |
|------|-----------------|
| `scalar type` | `tensor<f32>` |
| `1D tensor type` | `tensor<4xf32>` |
| `4D tensor type` | `tensor<1x8x8x16xf32>` |
| `i64 tensor type` | `tensor<2x3xi64>` |
| `bool tensor type` | `tensor<2x3xi1>` |
| `simple function` | `func.func @main(%arg0: ...) -> ...` wrapper |
| `module with return` | `return %0 : ...` terminator |
| `generic op form` | `"stablehlo.abs"(...)` quoted op name |
| `generic op with array attr` | `permutation = array<i64: 1, 0>` |

### 3.2 `test/Test/IR/PrettyOps.hs`

Tests custom pretty-printer formats for specific ops.

| Test | Expected Substring | Why Special |
|------|-------------------|-------------|
| `reduce` | `applies stablehlo.add across dimensions = [...]` | `applies` shorthand for reductions without regions |
| `convolution` | `dim_numbers = [...], window = {...}` | Inline dim_numbers + window attributes |
| `dot_general` | `batching_dims = [...], contracting_dims = [...]` | Inline batch/contract dims |
| `broadcast_in_dim` | `, dims = [...]` | Trailing dims suffix |
| `constant scalar` | `dense<3.0>` | No brackets for scalar |
| `constant 1D` | `dense<[1.0, 2.0, 3.0]>` | Bracket list for 1D |
| `constant 2D` | `dense<[[1.0, 2.0], [3.0, 4.0]]>` | Nested lists for 2D |
| `return` | `"stablehlo.return"` | Generic quoted form |
| `while` | `"stablehlo.while"` | Generic form with two regions |
| `if` | `"stablehlo.if"` | Generic form with two regions |
| `map` | `"stablehlo.map"` | Generic form with one region |

### 3.3 `test/Test/IR/PrettyNN.hs`

Tests pretty-printer output for NN layer ops.

| Test | Op | Expected Substring |
|------|-----|-------------------|
| `conv2d` | `conv2d` | `stablehlo.convolution` |
| `maxPool` | `maxPool` | `stablehlo.reduce_window` + `stablehlo.maximum` |
| `avgPool` | `avgPool` | `stablehlo.reduce_window` + `stablehlo.add` |
| `globalAvgPool` | `globalAvgPool` | `stablehlo.reduce` (two sequential single-dim reductions) |
| `softmax1D` | `softmax1D` | `stablehlo.exponential` + `stablehlo.divide` |
| `batchNormInference` | `batchNormInference` | `stablehlo.sqrt` + `stablehlo.subtract` + `stablehlo.divide` |
| `layerNorm` | `layerNorm` | `stablehlo.reduce` |
| `gelu` | `gelu` | `stablehlo.tanh` |

**Note:** `batchNormInference` and `gelu` are composite ops (implemented from primitives), so they do not emit dedicated StableHLO op names.

### 3.4 `test/Test/IR/PrettyControlFlow.hs`

Tests pretty-printer for control flow ops.

| Test | Op | Expected Substring |
|------|-----|-------------------|
| `while` | `while` | `"stablehlo.while"` + two regions |
| `if` | `conditional` | `"stablehlo.if"` + two regions |
| `compare` | `compare` | `"stablehlo.compare"` + `"LT"` |
| `sort` | `sort` | `"stablehlo.sort"` + comparator region |

### 3.5 `test/Test/IR/Builder.hs`

Tests the `Builder` monad state management.

| Test | What it verifies |
|------|-----------------|
| `value ids are sequential` | `%arg0`, `%arg1` for args; `%0`, `%1` for ops |
| `module has func.func wrapper` | `module { func.func @main(...) }` |
| `single result type in signature` | `-> tensor<3x4xf32>` |

### 3.6 `test/Test/EDSL/Ops.hs`

The largest golden test module. It exercises virtually every EDSL op and verifies that the rendered MLIR contains the expected StableHLO op name or structural pattern.

**Categories:**
- **Binary element-wise** (8 tests): `add`, `sub`, `multiply`, `divide`, `maximum`, `minimum`
- **Unary element-wise** (7 tests): `relu`, `negate`, `abs'`, `exponential`, `logarithm`, `tanh`, `erf`
- **Shape manipulation** (6 tests): `reshape`, `transpose`, `broadcastWithDims`, `concatenate`, `concatenate2`, `iota`
- **Matmul** (3 tests): `matmul`, `dotGeneral`, `linear`
- **Reductions** (4 tests): `reduceSum`, `reduceWindow`, `maxPool`, `avgPool`, `globalAvgPool`
- **NN layers** (8 tests): `conv2d`, `conv2dWithPadding`, `softmax1D`, `softmax2D`, `batchNormInference`, `layerNorm`, `gelu`, `transposeConvolution`
- **Control flow** (2 tests): `conditional`, `compare`
- **Data movement** (8 tests): `gather`, `scatter`, `slice`, `pad`, `dynamicSlice`, `select`, `convert`, `sort`, `map`
- **Constants** (3 tests): scalar, 2D, `i64`

---

## 4. Tier 2 — End-to-End Numerical Tests

These tests run the full pipeline: EDSL → AST → Pretty → PJRT compile → XLA codegen → CPU execution → buffer transfer back to host.

### 4.1 `test/Test/Runtime/EndToEndArithmetic.hs`

| Test | Op | Input A | Input B | Expected |
|------|-----|---------|---------|----------|
| `relu` | `relu` | `[1,2,3,4]` | — | `[1,2,3,4]` |
| `negate` | `negate` | `[1,2,3,4]` | — | `[-1,-2,-3,-4]` |
| `abs` | `abs'` | `[1,2,3,4]` | — | `[1,2,3,4]` |
| `exponential` | `exponential` | `[1,2,3,4]` | — | `[e¹,e²,e³,e⁴]` |
| `add` | `add` | `[1,2,3,4]` | `[5,6,7,8]` | `[6,8,10,12]` |
| `sub` | `sub` | `[1,2,3,4]` | `[5,6,7,8]` | `[-4,-4,-4,-4]` |
| `multiply` | `multiply` | `[1,2,3,4]` | `[5,6,7,8]` | `[5,12,21,32]` |
| `divide` | `divide` | `[1,2,3,4]` | `[5,6,7,8]` | `[0.2,0.333,0.428,0.5]` |
| `maximum` | `maximum` | `[-1,2,-3,4]` | `[0,0,0,0]` | `[0,2,0,4]` |
| `minimum` | `minimum` | `[-1,2,-3,4]` | `[0,0,0,0]` | `[-1,0,-3,0]` |

Uses `e2eTestF32_1arg` and `e2eTestF32_2arg` from `Test.Utils`.

### 4.2 `test/Test/Runtime/EndToEndMatmul.hs`

| Test | Op | Input Shapes | Expected |
|------|-----|-------------|----------|
| `matmul 2D` | `matmul` | `[2,3] × [3,2]` | `[22,28,49,64]` |
| `linear no bias` | `linear` | `[3] × [3,2] + [2]` | `[3.0,3.0]` |
| `linearBatched` | `linearBatched` | `[2,3] × [3,2] + [2]` | `[3.1,3.1,7.6,7.6]` |
| `dotGeneral 3D x 2D` | `dotGeneral` | `[1,2,3] × [3,2]` | `[3.0,3.0,7.5,7.5]` |

**Note on `dotGeneral`:** Uses empty batch dims `[] []` and contract dims `[2] [0]` to avoid the PJRT "duplicated dimension" error that occurs when a dimension appears in both batch and contract lists.

### 4.3 `test/Test/Runtime/EndToEndDataMovement.hs`

| Test | Op | Input | Expected |
|------|-----|-------|----------|
| `slice 1D` | `slice` | `[0,1,2,3,4]` | `[1,2,3]` |
| `slice with stride` | `slice` | `[0,1,2,3,4]` | `[0,2]` |
| `pad edge` | `pad` | `[1,2]` | `[0,1,2,0]` |
| `gather rows` | `gather` | `[3,4]` matrix | rows 0 and 0 |
| `select true` | `select` | two `[2,2]` + all-true pred | first operand |
| `select false` | `select` | two `[2,2]` + all-false pred | second operand |
| `convert f32 to f32` | `convert` | `[1,2]` | `[1,2]` |
| `conditional true` | `conditional` | scalar `true` | first branch |
| `conditional false` | `conditional` | scalar `false` | second branch |
| `map square` | `map` | `[1,2,3]` | `[1,4,9]` |
| `dynamicSlice` | `dynamicSlice` | `[0,1,2,3]` | `[1,2]` |

**PJRT Workarounds:**
- `select` and `conditional` require boolean predicates. The test uses `bufferTypePred` (not `bufferTypeU8`) when transferring `Word8` boolean vectors to device.
- `gather` requires all 7 attribute parameters (offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim, slice_sizes).

### 4.4 `test/Test/Runtime/EndToEndNN.hs`

| Test | Op | Input | Verification |
|------|-----|-------|-------------|
| `conv2d identity` | `conv2d` | `1×4×4×1`, zero kernel | all zeros |
| `softmax1D` | `softmax1D` | `[0,0,0]` | sums to 1.0, each ≈ 0.333 |
| `softmax2D` | `softmax2D` | `[2,3]` zeros | each row sums to 1.0 |
| `batchNorm identity` | `batchNormInference` | `1×2×2×2`, identity params | equals input |
| `gelu` | `gelu` | `[0,1,-1]` | `gelu(0)≈0`, `gelu(1)≈0.841`, `gelu(-1)≈-0.159` |
| `layerNorm` | `layerNorm` | `1×2×4` | uniform row has near-zero mean |
| `globalAvgPool` | `globalAvgPool` | `1×4×4×2` | channel 0 = 1.0, channel 1 = 2.0 |

**Important:** `globalAvgPool` uses two sequential single-dim `stablehlo.reduce` ops instead of one multi-dim reduce, because PJRT CPU v1.16.0 miscompiles multi-dimension `stablehlo.reduce` (see `doc/pjrt-cpu-v1160-parser-limitations.md`).

### 4.5 `test/Test/Runtime/EndToEndReductions.hs`

| Test | Op | Input | Expected |
|------|-----|-------|----------|
| `reduceSum all` | `reduceSum` | `[1..6]` shaped `[2,3]` | `21.0` |
| `maxPool 2x2` | `maxPool` | `[1..16]` shaped `[1,4,4,1]` | `[6,8,14,16]` |
| `avgPool 2x2` | `avgPool` | `[1..16]` shaped `[1,4,4,1]` | `[3.5,5.5,11.5,13.5]` |

### 4.6 `test/Test/Runtime/EndToEndShape.hs`

| Test | Op | Verification |
|------|-----|-------------|
| `reshape 2D to 1D` | `reshape` | `[1,2,3,4]` → `[1,2,3,4]` |
| `transpose` | `transpose` | `[[1,2],[3,4]]` → `[[1,3],[2,4]]` |
| `concatenate` | `concatenate` | `[1,2] + [3,4]` → `[1,2,3,4]` |
| `iota 1D` | `iota` + `convert` | `[0,1,2,3]` (iota returns `I64`, converted to `F32`) |

---

## 5. Tier 3 — Runtime Integration Tests

### 5.1 `test/Test/Runtime/Buffer.hs`

Tests buffer lifecycle and metadata queries.

| Test | What it verifies |
|------|-----------------|
| `buffer round-trip f32` | `toDeviceF32` → `fromDeviceF32` returns identical data |
| `buffer dimensions` | `bufferDimensions` matches the shape passed to `toDeviceF32` |
| `buffer element type f32` | `bufferElementType` returns `bufferTypeF32` |
| `buffer on-device size` | `bufferOnDeviceSize` is `> 0` |

### 5.2 `test/Test/Runtime/Async.hs`

Tests the async execution API.

| Test | What it verifies |
|------|-----------------|
| `buffer ready after sync execute` | `execute` (sync) produces buffers that `bufferReady` reports as ready |
| `await buffers` | `awaitBuffers` on ready buffers returns immediately |
| `executeAsync returns output` | `executeAsync` produces valid output buffers that can be read back |

### 5.3 `test/Test/Runtime/Errors.hs`

Tests error handling paths.

| Test | What it verifies |
|------|-----------------|
| `malformed mlir` | `compile` with invalid MLIR text throws `PJRTException` |

**Note:** The `invalid plugin path` test was removed because `c_pjrtLoadPlugin` behavior for nonexistent files varies by platform and C shim version.

---

## 6. Running the Test Suite

### Full suite
```bash
cabal test
```

### Specific test by pattern
```bash
cabal test --test-option='-p' --test-option='/matmul/'
cabal test --test-option='-p' --test-option='/EndToEnd.NN.softmax/'
```

### Just golden tests (no PJRT needed)
```bash
cabal test --test-option='-p' --test-option='/EDSL.Ops/ || /Pretty/ || /Builder/'
```

### Just E2E tests (requires PJRT CPU plugin)
```bash
cabal test --test-option='-p' --test-option='/EndToEnd/ || /Runtime.Buffer/ || /Runtime.Async/'
```

---

## 7. Adding a New Test

### 7.1 Golden Test (no PJRT needed)

Add to the appropriate `Test/IR/Pretty*.hs` or `Test/EDSL/Ops.hs`:

```haskell
, testCase "my new op" $ do
    let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
            [ FuncArg "arg0" (TensorType [2, 2] F32) ]
            $ do
                x <- arg @'[2, 2] @'F32
                y <- myOp x
                return y
    let rendered = render modu
    assertBool "contains stablehlo.my_op" $
        "stablehlo.my_op" `T.isInfixOf` rendered
```

### 7.2 E2E Numerical Test

Add to the appropriate `Test/Runtime/EndToEnd*.hs`:

```haskell
, testCase "my op on CPU" $ withPJRTCPU $ \api client -> do
    let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
            [ FuncArg "arg0" (TensorType [2, 2] F32) ]
            $ do
                x <- arg @'[2, 2] @'F32
                y <- myOp x
                return y
    exec <- compile api client (render modu)
    let inp = V.fromList [1.0, 2.0, 3.0, 4.0]
    bufIn <- toDeviceF32 api client inp [2, 2]
    [bufOut] <- execute api exec [bufIn]
    result <- fromDeviceF32 api bufOut 4
    result @?= V.fromList [expected1, expected2, expected3, expected4]
```

### 7.3 Registering the module

If you create a new test file:
1. Add it to `test/Main.hs` imports and `tests` list.
2. Add the module name to `other-modules` in `hhlo.cabal` under the `hhlo-test` stanza.
3. Add any new `build-depends` if needed.

---

## 8. Known PJRT Limitations Affecting Tests

The test suite documents and works around several PJRT CPU v1.16.0 parser limitations. See `doc/pjrt-cpu-v1160-parser-limitations.md` for the full catalog. Key ones relevant to tests:

| Limitation | Impact | Workaround in Tests |
|------------|--------|---------------------|
| `stablehlo.compare` custom form unparseable | Cannot test `compare` E2E | Golden test only |
| `stablehlo.sort` unparseable | Cannot test `sort` E2E | Golden test only |
| `stablehlo.reduce` multi-dim partial reduce miscompiled | `globalAvgPool` gave wrong answers | Two sequential single-dim reductions |
| `stablehlo.tuple` unparseable | Cannot test tuple return E2E | Golden test only |
| Integer constants must not have `.0` suffix | `gather`, `dynamicSlice` failed to compile | `formatDenseVal` emits `0` instead of `0.0` for integer dtypes |

---

## 9. Pretty-Printer Evolution: `applies` → Generic Region

Originally, `stablehlo.reduce` was printed with an `applies` shorthand:

```mlir
%0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0]
    : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
```

This worked for full reductions but caused PJRT to **ignore dimensions for partial reductions**, producing incorrect numerical results. The library was updated to emit the **generic region form**:

```mlir
%0 = "stablehlo.reduce"(%arg0, %cst) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
}) {dimensions = array<i64: 0>} : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
```

This is valid StableHLO that PJRT parses correctly for both full and partial reductions. The pretty-printer now handles both forms:
- If `applies` attribute is present → emit the custom shorthand (backward compat).
- If a region is present → emit the generic form.

The test suite validates both: golden tests check the rendered text, E2E tests verify the numerical result.
