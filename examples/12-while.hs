{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 12: While loop — count from 0 to 5.
--
-- NOTE: This example emits valid StableHLO MLIR, but the PJRT CPU plugin
-- v1.16.0 cannot parse 'stablehlo.compare' (required for the loop condition).
-- It is provided for MLIR inspection and will work on newer PJRT plugins
-- or GPU backends.
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-while

module Main where

import qualified Data.Text as T

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Builder
import HHLO.IR.Pretty

main :: IO ()
main = do
    putStrLn "=== Example 12: While Loop (MLIR print-only) ==="
    putStrLn "NOTE: PJRT CPU v1.16.0 cannot parse stablehlo.compare inside while."
    putStrLn "The emitted MLIR is valid and will execute on newer plugins/GPU.\n"

    let program = do
            initVal <- constant @'[] @'F32 0.0
            limit   <- constant @'[] @'F32 5.0
            one     <- constant @'[] @'F32 1.0

            result <- whileLoop initVal
                (\loopVar -> lessThan loopVar limit)
                (\loopVar -> do
                    let (Tensor v) = loopVar
                        (Tensor o) = one
                    add (Tensor v) (Tensor o))

            return result

    let modu = moduleFromBuilder @'[] @'F32 "main" [] program

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)
    putStrLn "\nExpected result on a compatible backend: [5.0]"
