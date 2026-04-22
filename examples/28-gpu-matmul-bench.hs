{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Vector.Storable as V
import Data.Int (Int64)
import Data.Time.Clock (diffUTCTime, getCurrentTime)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
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
    putStrLn $ "Matrix multiply benchmark: " ++ show matmulSize ++ "x" ++ show matmulSize

    mDev <- defaultGPUDevice api client
    dev <- maybe (error "No GPU found") return mDev

    let modu = moduleFromBuilder @'[4096, 4096] @'F32 "main"
            [ FuncArg "arg0" (TensorType [4096, 4096] F32)
            , FuncArg "arg1" (TensorType [4096, 4096] F32)
            ]
            $ do
                x <- arg @'[4096, 4096] @'F32
                y <- arg @'[4096, 4096] @'F32
                z <- matmul x y
                return z

    putStrLn "Compiling..."
    exec <- compile api client (render modu)

    let n = matmulSize * matmulSize
        inputA = V.replicate n (1.0 :: Float)
        inputB = V.replicate n (1.0 :: Float)
        dims = [fromIntegral matmulSize, fromIntegral matmulSize] :: [Int64]

    putStrLn "Uploading to GPU..."
    bufA <- toDeviceOn api client dev inputA dims bufferTypeF32
    bufB <- toDeviceOn api client dev inputB dims bufferTypeF32

    putStrLn "Executing on GPU..."
    t0 <- getCurrentTime
    [bufOut] <- executeOn api exec dev [bufA, bufB]
    result <- fromDeviceF32 api bufOut n
    t1 <- getCurrentTime

    let gpuTime = realToFrac (diffUTCTime t1 t0) :: Double
    putStrLn $ "GPU time (including D2H): " ++ show gpuTime ++ "s"

    -- Verify a corner element: each output element is sum of 4096 * 1.0 * 1.0 = 4096.0
    let corner = result V.! 0
    putStrLn $ "Corner element: " ++ show corner ++ " (expected 4096.0)"

    if abs (corner - 4096.0) < 0.1
        then putStrLn "Verification PASSED"
        else putStrLn "Verification FAILED"
