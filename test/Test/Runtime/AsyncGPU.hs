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
import HHLO.Runtime.Compile
import HHLO.Runtime.Buffer
import qualified HHLO.Runtime.Async as Async
import Test.Runtime.GPUResource (GPUResource(..))
import Test.Utils (toDeviceF32On)

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "Runtime.AsyncGPU"
    [ testCase "gpu executeAsync + await" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 2, Just 2] F32)
                , FuncArg "arg1" (TensorType [Just 2, Just 2] F32)
                ]
                $ do
                    x <- arg
                    y <- arg
                    z <- add x y
                    return z
        exec <- compile api client (render modu)
        let inputA = V.fromList [1, 2, 3, 4] :: V.Vector Float
            inputB = V.fromList [10, 20, 30, 40] :: V.Vector Float
        bufA <- toDeviceF32On api client dev inputA [2, 2]
        bufB <- toDeviceF32On api client dev inputB [2, 2]
        [bufOut] <- Async.executeAsync api exec [bufA, bufB]
        Async.awaitBuffers api [bufOut]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [11, 22, 33, 44]

    , testCase "gpu buffer ready poll" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 2, Just 2] F32)
                ]
                $ do
                    x <- arg
                    y <- relu x
                    return y
        exec <- compile api client (render modu)
        let input = V.fromList [-1, 2, -3, 4] :: V.Vector Float
        bufIn <- toDeviceF32On api client dev input [2, 2]
        [bufOut] <- Async.executeAsync api exec [bufIn]
        let loop = do
                ready <- Async.bufferReady api bufOut
                if ready then return () else loop
        loop
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [0, 2, 0, 4]
    ]
