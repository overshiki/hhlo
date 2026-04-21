{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 19: Sort a 1D tensor.
--
-- Input: [3.0, 1.0, 4.0, 1.0, 5.0]
-- Output: [1.0, 1.0, 3.0, 4.0, 5.0] (ascending, stable)
--
-- NOTE: This example emits valid StableHLO MLIR, but the PJRT CPU plugin
-- v1.16.0 cannot parse 'stablehlo.compare' (required for the sort comparator).
-- It is provided for MLIR inspection and will work on newer PJRT plugins
-- or GPU backends.
--
-- Build and run with:
--   cabal run example-sort

module Main where

import qualified Data.Text as T

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty

main :: IO ()
main = do
    putStrLn "=== Example 19: Sort (MLIR print-only) ==="
    putStrLn "NOTE: PJRT CPU v1.16.0 cannot parse stablehlo.compare inside sort comparator."
    putStrLn "The emitted MLIR is valid and will execute on newer plugins/GPU.\n"

    let modu = moduleFromBuilder @'[5] @'F32 "main"
            [ FuncArg "operand" (TensorType [5] F32)
            ]
            $ do
                operand <- arg @'[5] @'F32
                sort operand 0 True $ \a b -> do
                    lessThan a b

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)
    putStrLn "\nExpected result on a compatible backend: [1.0, 1.0, 3.0, 4.0, 5.0]"
