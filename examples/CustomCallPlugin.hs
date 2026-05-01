{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example: using a custom-call plugin from HHLO.
--
-- Before running this example you must compile the CUDA kernel:
--
-- @
--   cd examples/cbits && bash build.sh
-- @
--
-- Then run with the GPU plugin:
--
-- @
--   cabal run example-custom-call --flag=examples
-- @
module Main where

import qualified Data.Vector.Storable as V
import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.CustomCall (loadCustomCallLibrary)
import HHLO.Session

main :: IO ()
main = withGPU $ \sess -> do
    -- 1. Load the custom-call library into the global symbol namespace.
    --    This MUST happen before 'compile'.
    loadCustomCallLibrary "examples/cbits/libvector_add.so"

    -- 2. Build a StableHLO module that uses the custom call.
    let modu = moduleFromBuilder @'[4] @'F32 "vector_add"
            [ FuncArg "a" (TensorType [4] F32)
            , FuncArg "b" (TensorType [4] F32)
            ]
            $ do
                a <- arg @'[4] @'F32
                b <- arg @'[4] @'F32
                c <- customCall1 "vector_add" [a, b] "" False
                return c

    putStrLn "=== Emitted MLIR ==="
    putStrLn (render modu)

    -- 3. Compile and execute via PJRT.
    compiled <- compile sess modu
    let aVals = V.fromList [1.0, 2.0, 3.0, 4.0]
        bVals = V.fromList [10.0, 20.0, 30.0, 40.0]
    result <- run sess compiled (aVals, bVals)

    putStrLn "=== Result ==="
    print result
