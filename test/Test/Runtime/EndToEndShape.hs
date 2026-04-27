{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndShape where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer
import Test.Utils

input2x2 :: V.Vector Float
input2x2 = V.fromList [1.0, 2.0, 3.0, 4.0]

tests :: TestTree
tests = testGroup "EndToEnd.Shape"
    [ testCase "reshape flatten" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32) ]
                $ do
                    x <- arg
                    y <- reshape @'[2, 2] @'[4] x
                    return y
        exec <- compile api client (render modu)
        bufIn <- toDeviceF32 api client input2x2 [2, 2]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 4
        result @?= input2x2
    , testCase "transpose swap" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32) ]
                $ do
                    x <- arg
                    y <- transpose @'[2, 2] @'[2, 2] [1, 0] x
                    return y
        exec <- compile api client (render modu)
        bufIn <- toDeviceF32 api client input2x2 [2, 2]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [1.0, 3.0, 2.0, 4.0]
    , testCase "transpose identity" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32) ]
                $ do
                    x <- arg
                    y <- transpose @'[2, 2] @'[2, 2] [0, 1] x
                    return y
        exec <- compile api client (render modu)
        bufIn <- toDeviceF32 api client input2x2 [2, 2]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 4
        result @?= input2x2
    , testCase "broadcast scalar" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main" [] $ do
                x <- constant @'[] @'F32 5.0
                y <- broadcastWithDims @'[] @'[2, 2] [] x
                return y
        exec <- compile api client (render modu)
        [bufOut] <- execute api exec []
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [5.0, 5.0, 5.0, 5.0]
    , testCase "concatenate" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32)
                , FuncArg "arg1" (TensorType [2] F32)
                ]
                $ do
                    x <- arg
                    y <- arg
                    z <- concatenate @'[2] @'[4] 0 [x, y]
                    return z
        exec <- compile api client (render modu)
        bufA <- toDeviceF32 api client (V.fromList [1.0, 2.0]) [2]
        bufB <- toDeviceF32 api client (V.fromList [3.0, 4.0]) [2]
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [1.0, 2.0, 3.0, 4.0]
    , testCase "iota 1D" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[4] @'F32 "main" [] $ do
                x <- iota @'[4] 0
                y <- convert @'[4] @'I64 @'F32 x
                return y
        exec <- compile api client (render modu)
        [bufOut] <- execute api exec []
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [0.0, 1.0, 2.0, 3.0]
    , testCase "split" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [4] F32) ]
                $ do
                    x <- arg @'[4] @'F32
                    ys <- split @'[4] @'[2] 0 2 x
                    case ys of
                        (y1:_) -> return y1
                        _ -> error "expected at least one split"
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0]
        bufIn <- toDeviceF32 api client inp [4]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [1.0, 2.0]
    , testCase "stack" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32)
                , FuncArg "arg1" (TensorType [2] F32)
                ]
                $ do
                    x <- arg @'[2] @'F32
                    y <- arg @'[2] @'F32
                    z <- stack @'[2] @'[2, 2] 0 [x, y]
                    return z
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0]
            b = V.fromList [3.0, 4.0]
        bufA <- toDeviceF32 api client a [2]
        bufB <- toDeviceF32 api client b [2]
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [1.0, 2.0, 3.0, 4.0]
    ]
