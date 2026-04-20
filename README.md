# HHLO — Haskell Frontend for StableHLO

HHLO is a Haskell library and runtime for building, compiling, and executing machine learning programs targeting [StableHLO](https://github.com/openxla/stablehlo), the portable, versioned intermediate representation of the [OpenXLA](https://openxla.org/) ecosystem.

Instead of replicating JAX's Python-based tracing infrastructure, HHLO generates StableHLO MLIR text directly from Haskell and compiles it to CPU (and eventually GPU/TPU) via the [PJRT](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) plugin interface.

---

## Design

HHLO is structured in four layers:

```
┌─────────────────────────────────────┐
│  EDSL (HHLO.EDSL.Ops)               │  Type-safe frontend: add, matmul, etc.
├─────────────────────────────────────┤
│  IR Builder (HHLO.IR.Builder)       │  Stateful monad for constructing MLIR
├─────────────────────────────────────┤
│  Pretty Printer (HHLO.IR.Pretty)    │  Emits StableHLO MLIR text
├─────────────────────────────────────┤
│  PJRT Runtime (HHLO.Runtime.*)      │  Compile → Execute on device
└─────────────────────────────────────┘
```

**Text Emission + PJRT**

The library emits StableHLO MLIR text directly and hands it to `PJRT_Client_Compile`. This is the same path used by JAX's C++ backend and avoids the heavy dependency of building LLVM/MLIR from source.

**ForeignPtr Finalizers**

PJRT buffers and executables are managed by `ForeignPtr` finalizers that automatically call `PJRT_Buffer_Destroy` and `PJRT_LoadedExecutable_Destroy` when values are garbage-collected. You can still let references drop out of scope without explicit cleanup.

**Dynamic Output Counts**

The runtime queries the compiled executable for its actual number of outputs via `PJRT_Executable_NumOutputs` instead of guessing or hardcoding a maximum.

---

## Installation

### System Requirements

- GHC 9.6+ and Cabal 3.10+
- Linux x86_64 (other platforms supported by PJRT artifacts may work)
- `curl`, `tar`, and standard C toolchain (`gcc` or `clang`)
- `libstdc++` and `libdl` (usually present on Linux)

### Download PJRT Plugins

Run the provided script to download prebuilt PJRT CPU plugin(s):

```bash
./pjrt_script.sh
```

This downloads `libpjrt_cpu.so` from the [zml/pjrt-artifacts](https://github.com/zml/pjrt-artifacts) nightly builds into `deps/pjrt/`. If you have an NVIDIA GPU with `nvidia-smi` available, the CUDA plugin is also fetched automatically.

### Build the Project

```bash
cabal build all
```

This compiles the library, the demo, the examples, and the test suite.

---

## Usage

### EDSL Quick Start

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import qualified Data.Text as T

-- Build a program: c = a + b
program :: Module
program = moduleFromBuilder @'[2,2] @'F32 "main"
    [ FuncArg "a" (TensorType [2, 2] F32)
    , FuncArg "b" (TensorType [2, 2] F32)
    ]
    $ do
        a <- arg
        b <- arg
        c <- add a b
        return c

main :: IO ()
main = T.putStrLn (render program)
```

Output:
```mlir
module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
      %0 = stablehlo.add %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
      return %0 : tensor<2x2xf32>
  }
}
```

### Running the Demo

```bash
LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run hhlo-demo
```

The demo builds a `stablehlo.add` program via the EDSL, compiles it with PJRT CPU, creates F32 input buffers, executes, and reads back the result:

```
=== HHLO End-to-End Demo ===
Loading PJRT CPU plugin...
Plugin loaded.
...
Result: [6.0,8.0,10.0,12.0]
SUCCESS: Results match expected values!
```

### Running Examples

Three standalone examples are provided in `examples/`:

| Example | Command | Description |
|---------|---------|-------------|
| Element-wise add | `cabal run example-add` | `c = a + b` on 2×2 matrices |
| Matrix multiply | `cabal run example-matmul` | 2×3 @ 3×2 matmul |
| Chained ops | `cabal run example-chain-ops` | `(a + b) * (a - b)` |

All examples must be run with `LD_LIBRARY_PATH` pointing to the PJRT plugins:

```bash
export LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH
cabal run example-add
cabal run example-matmul
cabal run example-chain-ops
```

---

## Tests

```bash
cabal test
```

The test suite includes:

- **Pretty printer tests** — Verify MLIR text generation for types, functions, and modules.
- **End-to-end runtime test** — Load the CPU plugin, compile a StableHLO `add` program, execute it, and verify the numerical result.

Sample output:
```
HHLO Tests
  Pretty
    scalar type:            OK
    2D tensor type:         OK
    simple function:        OK
  EndToEnd
    add two tensors on CPU: OK (0.03s)

All 4 tests passed (0.03s)
```

---

## Project Structure

```
.
├── app/                    # hhlo-demo executable
├── cbits/                  # C shim around PJRT C API
│   └── pjrt_shim.c
├── deps/
│   └── pjrt/               # Downloaded PJRT plugins (.so files)
├── doc/                    # Architecture and design documents
├── examples/               # Standalone example programs
│   ├── 01-add.hs
│   ├── 02-matmul.hs
│   └── 03-chain-ops.hs
├── src/HHLO/
│   ├── Core/Types.hs       # DType, Shape, HostType type families
│   ├── IR/
│   │   ├── AST.hs          # MLIR AST (Operation, Function, Module)
│   │   ├── Builder.hs      # Stateful Builder monad
│   │   └── Pretty.hs       # MLIR text pretty-printer
│   ├── EDSL/Ops.hs         # Type-safe frontend ops
│   └── Runtime/
│       ├── PJRT/
│       │   ├── FFI.hs      # C FFI declarations
│       │   ├── Types.hs    # Opaque pointer newtypes + buffer type constants
│       │   └── Error.hs    # PJRT error handling
│       ├── Compile.hs      # MLIR → PJRT executable
│       ├── Execute.hs      # Synchronous execution
│       └── Buffer.hs       # Host↔device buffer transfers
├── test/
│   ├── Test/IR/Pretty.hs
│   └── Test/Runtime/EndToEnd.hs
├── hhlo.cabal
├── pjrt_script.sh
└── README.md
```

---

## Architecture Docs

The `doc/` directory contains detailed design documents:

| Document | Contents |
|----------|----------|
| `design.md` | High-level system design and goals |
| `implementation-design.md` | Feasibility analysis and roadmap |
| `understanding-pjrt.md` | Deep dive into PJRT C API vtable patterns |
| `understanding-zml-pjrt-artifacts.md` | Prebuilt artifact sources and versioning |
| `text-emission-vs-mlir-hs.md` | Why text emission was chosen over `mlir-hs` |
| `progress-and-remaining-work.md` | Current status and backlog |

---

## License

MIT License — see [LICENSE](LICENSE) (if present) or contact the maintainer.
