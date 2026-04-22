{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.Async where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute hiding (executeAsync)
import HHLO.Runtime.Buffer
import qualified HHLO.Runtime.Async as Async
import Test.Utils

tests :: TestTree
tests = testGroup "Runtime.Async"
    [ testCase "buffer ready after sync execute" $ withPJRTCPU $ \api client -> do
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
        let a = V.fromList [1.0, 2.0, 3.0, 4.0]
            b = V.fromList [5.0, 6.0, 7.0, 8.0]
        bufA <- toDeviceF32 api client a [2, 2]
        bufB <- toDeviceF32 api client b [2, 2]
        [bufOut] <- execute api exec [bufA, bufB]
        ready <- Async.bufferReady api bufOut
        ready @?= True
    , testCase "await buffers" $ withPJRTCPU $ \api client -> do
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
        let a = V.fromList [1.0, 2.0, 3.0, 4.0]
            b = V.fromList [5.0, 6.0, 7.0, 8.0]
        bufA <- toDeviceF32 api client a [2, 2]
        bufB <- toDeviceF32 api client b [2, 2]
        [bufOut] <- execute api exec [bufA, bufB]
        Async.awaitBuffers api [bufOut]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [6.0, 8.0, 10.0, 12.0]
    , testCase "executeAsync returns output" $ withPJRTCPU $ \api client -> do
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
        let a = V.fromList [1.0, 2.0, 3.0, 4.0]
            b = V.fromList [5.0, 6.0, 7.0, 8.0]
        bufA <- toDeviceF32 api client a [2, 2]
        bufB <- toDeviceF32 api client b [2, 2]
        [bufOut] <- Async.executeAsync api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [6.0, 8.0, 10.0, 12.0]
    ]
