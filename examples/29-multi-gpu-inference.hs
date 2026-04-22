{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Vector.Storable as V
import Data.Int (Int64)
import Data.Time.Clock (diffUTCTime, getCurrentTime)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..), Module)
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Device
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

matmulSize :: Int
matmulSize = 4096

main :: IO ()
main = withPJRTGPU $ \api client -> do
    devs <- addressableDevices api client
    let numDevs = length devs
    putStrLn $ "Discovered " ++ show numDevs ++ " GPU(s)"

    -- Build a 4096×4096 matmul program
    let modu :: Module
        modu = moduleFromBuilder @'[4096, 4096] @'F32 "main"
            [ FuncArg "arg0" (TensorType [4096, 4096] F32)
            , FuncArg "arg1" (TensorType [4096, 4096] F32)
            ]
            $ do
                x <- arg @'[4096, 4096] @'F32
                y <- arg @'[4096, 4096] @'F32
                z <- matmul x y
                return z

    putStrLn "Compiling (with num_replicas = num_devices)..."
    exec <- compileWithOptions api client (render modu)
                (defaultCompileOptions { optNumReplicas = numDevs })

    let n = matmulSize * matmulSize
        dims = [fromIntegral matmulSize, fromIntegral matmulSize] :: [Int64]
        -- Input A: all 1.0s
        inputA = V.replicate n (1.0 :: Float)
        -- Input B: all 1.0s → each output element = sum of 4096 * 1.0 * 1.0 = 4096.0
        inputB = V.replicate n (1.0 :: Float)

    -- Upload inputs to every GPU
    putStrLn $ "Uploading inputs to " ++ show numDevs ++ " GPU(s)..."
    deviceArgs <- mapM (\dev -> do
        bufA <- toDeviceOn api client dev inputA dims bufferTypeF32
        bufB <- toDeviceOn api client dev inputB dims bufferTypeF32
        return (dev, [bufA, bufB])
      ) devs

    -- Warm-up run
    putStrLn "Warm-up..."
    _ <- executeReplicas api exec deviceArgs

    -- Timed run
    putStrLn "Running inference on all GPUs concurrently..."
    t0 <- getCurrentTime
    results <- executeReplicas api exec deviceArgs
    t1 <- getCurrentTime

    let elapsed = realToFrac (diffUTCTime t1 t0) :: Double
    putStrLn $ "All " ++ show numDevs ++ " GPU(s) completed in " ++ show elapsed ++ "s"

    -- Verify results from each GPU
    putStrLn "Verifying results..."
    allCorrect <- and <$> mapM (\(idx, outs) -> do
        let [bufOut] = outs
        result <- fromDeviceF32 api bufOut n
        let corner = result V.! 0
        let expected = 4096.0 :: Float
        let ok = abs (corner - expected) < 0.1
        putStrLn $ "  GPU " ++ show idx ++ " corner = " ++ show corner
                   ++ if ok then " ✓" else " ✗ FAILED"
        return ok
      ) (zip [0..] results)

    if allCorrect
        then putStrLn "Multi-GPU inference PASSED!"
        else putStrLn "Multi-GPU inference FAILED!"
