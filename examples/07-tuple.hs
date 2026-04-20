{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 7: Multi-result function using the general 'Tuple' type.
--
-- Returns both the sum and the product of two input tensors.
--
-- Note: The generated MLIR uses a multi-result 'func.func' signature which
-- is valid StableHLO, but the PJRT CPU plugin (v1.16.0) does not yet parse
-- multi-result functions.  This example prints the generated MLIR for
-- inspection only.
--
-- Build and run with:
--   cabal run example-tuple

module Main where

import qualified Data.Text as T
import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty

main :: IO ()
main = do
    putStrLn "=== Example 7: Multi-Result Tuple (MLIR only) ==="

    -- Build a function that returns two results: (a + b, a * b)
    let modu = moduleFromBuilderT @'[ '[2, 2], '[2, 2] ] @'[ 'F32, 'F32 ] "main"
            [ FuncArg "a" (TensorType [2, 2] F32)
            , FuncArg "b" (TensorType [2, 2] F32)
            ]
            $ do
                a <- arg @'[2, 2] @'F32
                b <- arg @'[2, 2] @'F32
                s <- add a b
                p <- multiply a b
                returnT (s ::: p ::: TNil)

    putStrLn "Generated MLIR (multi-result func.func):"
    putStrLn (T.unpack $ render modu)
    putStrLn ""
    putStrLn "Note: Multi-result func.func requires a PJRT plugin that supports"
    putStrLn "      StableHLO multi-result parsing. The CPU plugin v1.16.0 does"
    putStrLn "      not yet support this syntax."
