{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.AsyncGPU (tests) where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Device
import HHLO.Runtime.Compile
import HHLO.Runtime.Buffer
import HHLO.Runtime.Async

tests :: TestTree
tests = testGroup "Runtime.AsyncGPU"
    [ testCase "gpu executeAsync + await" gpuExecuteAsyncAwait
    , testCase "gpu buffer ready poll" gpuBufferReadyPoll
    ]

gpuExecuteAsyncAwait :: IO ()
gpuExecuteAsyncAwait = withPJRTGPU $ \api client -> do
    mDev <- defaultGPUDevice api client
    dev <- maybe (assertFailure "No GPU found") return mDev

    let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
            [ FuncArg "arg0" (TensorType [2, 2] F32)
            , FuncArg "arg1" (TensorType [2, 2] F32)
            ]
            $ do
                x <- arg
                y <- arg
                z <- add x y
                return z

    exec <- compile api client (render modu)

    let inputA = V.fromList [1, 2, 3, 4] :: V.Vector Float
        inputB = V.fromList [10, 20, 30, 40] :: V.Vector Float

    bufA <- toDeviceOn api client dev inputA [2, 2] bufferTypeF32
    bufB <- toDeviceOn api client dev inputB [2, 2] bufferTypeF32

    -- executeAsync should return immediately
    [bufOut] <- executeAsync api exec [bufA, bufB]

    -- await the output buffer
    awaitBuffers api [bufOut]

    result <- fromDeviceF32 api bufOut 4
    result @?= V.fromList [11, 22, 33, 44]

gpuBufferReadyPoll :: IO ()
gpuBufferReadyPoll = withPJRTGPU $ \api client -> do
    mDev <- defaultGPUDevice api client
    dev <- maybe (assertFailure "No GPU found") return mDev

    let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
            [ FuncArg "arg0" (TensorType [2, 2] F32)
            ]
            $ do
                x <- arg
                y <- relu x
                return y

    exec <- compile api client (render modu)

    let input = V.fromList [-1, 2, -3, 4] :: V.Vector Float
    bufIn <- toDeviceOn api client dev input [2, 2] bufferTypeF32

    [bufOut] <- executeAsync api exec [bufIn]

    -- Poll until ready (should become ready quickly for tiny ops)
    let loop = do
            ready <- bufferReady api bufOut
            if ready then return () else loop
    loop

    result <- fromDeviceF32 api bufOut 4
    result @?= V.fromList [0, 2, 0, 4]
