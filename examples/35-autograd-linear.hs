{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 35: Autograd — Gradient of a linear model + MSE loss.
--
-- We define:  y = W * x + b   (with fixed W, b)
-- and loss:   L = (y - target)²
--
-- grad_L w.r.t. x = 2 * W * (y - target)
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-autograd-linear

module Main where

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Builder (Tensor, Builder)
import HHLO.Session
import HHLO.Autograd

main :: IO ()
main = withCPU $ \sess -> do
    putStrLn "=== Example 35: Autograd Linear + MSE ==="
    putStrLn "y = W*x + b,  L = (y - target)^2"
    putStrLn "grad_x L = 2 * W * (y - target)"
    putStrLn ""

    -- Fixed parameters: W = 2.0, b = 1.0, target = 5.0
    -- f(x) = (2*x + 1 - 5)^2 = (2*x - 4)^2
    -- grad f(x) = 2 * 2 * (2*x - 4) = 8*x - 16
    -- At x = 3.0: grad = 8*3 - 16 = 8
    let f :: Tensor '[1] 'F32 -> Builder (Tensor '[] 'F32)
        f x = do
            w    <- constant @'[1] @'F32 2.0
            b    <- constant @'[1] @'F32 1.0
            tgt  <- constant @'[1] @'F32 5.0
            wx   <- multiply w x
            y    <- add wx b
            diff <- sub y tgt
            sq   <- multiply diff diff
            sumAll sq

    let gradMod = gradModule @'[1] @'F32 f

    putStrLn "Generated gradient MLIR:"
    print gradMod
    putStrLn ""

    compiled <- compile sess gradMod
    let input = hostFromList @'[1] @'F32 [3.0]
    result <- run sess compiled input :: IO (HostTensor '[1] 'F32)

    putStrLn $ "Input x:      " ++ show (hostToList result)
    putStrLn $ "Expected grad: [8.0]  (analytical: 8*3 - 16)"

    let resultList :: [Float]
        resultList = hostToList result
    if abs (head resultList - 8.0) < 0.001
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"
