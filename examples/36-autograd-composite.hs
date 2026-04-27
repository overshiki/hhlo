{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 36: Autograd through a composite function (ReLU + linear + sum).
--
-- f(x) = sum( relu( w ⊙ x + b ) )
--
-- Demonstrates that autograd works through non-linear ops
-- (max/relu, multiply, add) and reduce (sumAll).
--
-- With w = [2, 3], b = [1, 1], x = [1, 1]:
--   z = w⊙x + b = [3, 4]
--   relu(z) = [3, 4]
--   f(x) = 7
--
-- grad w.r.t. x:
--   d(relu(z))/dz = [1, 1]  (both positive)
--   dx = w ⊙ [1, 1] = [2, 3]
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-autograd-composite

module Main where

import Prelude hiding (maximum)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Builder (Tensor, Builder)
import HHLO.Session
import HHLO.Autograd

main :: IO ()
main = withCPU $ \sess -> do
    putStrLn "=== Example 36: Autograd Composite (ReLU + Linear + Sum) ==="
    putStrLn "f(x) = sum( relu( w * x + b ) )"
    putStrLn ""

    let f :: Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
        f x = do
            w    <- constant @'[2] @'F32 1.0
            b    <- constant @'[2] @'F32 1.0
            wx   <- multiply w x
            z    <- add wx b
            zero <- constant @'[2] @'F32 0.0
            reluZ <- maximum z zero
            sumAll reluZ

    let gradMod = gradModule @'[2] @'F32 f

    putStrLn "Generated gradient MLIR:"
    print gradMod
    putStrLn ""

    compiled <- compile sess gradMod
    let input = hostFromList @'[2] @'F32 [1.0, 1.0]
    result <- run sess compiled input :: IO (HostTensor '[2] 'F32)

    putStrLn $ "Input x:       " ++ show (hostToList result)
    putStrLn $ "Expected grad: [1.0, 1.0]  (w=[1,1], b=[1,1])"

    let res :: [Float]
        res = hostToList result
    if all (\v -> abs (v - 1.0) < 0.001) res
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"
