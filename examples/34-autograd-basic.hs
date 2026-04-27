{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 34: Basic Autograd — Gradient of sum(x²).
--
-- Demonstrates the simplest use of HHLO.Autograd:
--   f(x) = sum(x * x)   =>   grad f(x) = 2 * x
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-autograd-basic

module Main where

import Prelude hiding (maximum)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Builder (Tensor, Builder)
import HHLO.Session
import HHLO.Autograd

main :: IO ()
main = withCPU $ \sess -> do
    putStrLn "=== Example 34: Basic Autograd ==="
    putStrLn "f(x) = sum(x * x)  =>  grad f(x) = 2 * x"
    putStrLn ""

    -- Define a scalar-valued function.
    let f :: Tensor '[3] 'F32 -> Builder (Tensor '[] 'F32)
        f x = do
            sq <- multiply x x
            sumAll sq

    -- Compute its gradient as a standalone Module.
    let gradMod = gradModule @'[3] @'F32 f

    putStrLn "Generated gradient MLIR:"
    print gradMod
    putStrLn ""

    -- Compile and run.
    compiled <- compile sess gradMod
    let input = hostFromList @'[3] @'F32 [1.0, 2.0, 3.0]
    result <- run sess compiled input :: IO (HostTensor '[3] 'F32)

    putStrLn $ "Input:    " ++ show (hostToList result)
    putStrLn $ "Expected: [2.0, 4.0, 6.0]"

    let resultList :: [Float]
        resultList = hostToList result
    if resultList == [2.0, 4.0, 6.0]
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"
