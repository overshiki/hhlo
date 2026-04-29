{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndNNGPU (tests) where

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
import HHLO.Runtime.PJRT.Types (bufferTypeF32, bufferTypePred, bufferTypeS64)
import Test.Utils
import Test.Runtime.GPUResource (GPUResource(..))

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "EndToEnd.NNGPU"
    [ testCase "conv2d identity" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                [ FuncArg "arg0" (TensorType [1, 4, 4, 1] F32) ]
                $ do
                    x <- arg @'[1, 4, 4, 1] @'F32
                    k <- constant @'[3, 3, 1, 1] @'F32 0.0
                    y <- conv2d x k
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1..16]
        bufIn <- toDeviceF32On api client dev inp [1, 4, 4, 1]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 4
        V.all (== 0.0) result @? "conv2d with zero kernel should output zeros"
    , testCase "softmax1D" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[3] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3] F32) ]
                $ do
                    x <- arg
                    y <- softmax1D x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 0.0, 0.0]
        bufIn <- toDeviceF32On api client dev inp [3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 3
        let expected = V.fromList [1/3, 1/3, 1/3]
        assertBool "softmax sums to 1" $ abs (V.sum result - 1.0) < 0.01
        assertBool "softmax values close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "softmax2D" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2, 3] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 3] F32) ]
                $ do
                    x <- arg
                    y <- softmax2D x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bufIn <- toDeviceF32On api client dev inp [2, 3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 6
        assertBool "softmax2D row 0 sums to 1" $ abs (V.sum (V.slice 0 3 result) - 1.0) < 0.01
        assertBool "softmax2D row 1 sums to 1" $ abs (V.sum (V.slice 3 3 result) - 1.0) < 0.01
    , testCase "batchNorm identity" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[1, 2, 2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [1, 2, 2, 2] F32) ]
                $ do
                    x <- arg @'[1, 2, 2, 2] @'F32
                    s <- constant @'[2] @'F32 1.0
                    o <- constant @'[2] @'F32 0.0
                    m <- constant @'[2] @'F32 0.0
                    v <- constant @'[2] @'F32 1.0
                    y <- batchNormInference x s o m v
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        bufIn <- toDeviceF32On api client dev inp [1, 2, 2, 2]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 8
        assertBool "batchNorm identity" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList inp))
    , testCase "gelu" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[3] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3] F32) ]
                $ do
                    x <- arg @'[3] @'F32
                    y <- gelu x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 1.0, -1.0]
        bufIn <- toDeviceF32On api client dev inp [3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 3
        assertBool "gelu(0) ≈ 0" $ abs (result V.! 0) < 0.01
        assertBool "gelu(1) ≈ 0.84" $ abs (result V.! 1 - 0.841) < 0.01
        assertBool "gelu(-1) ≈ -0.16" $ abs (result V.! 2 + 0.159) < 0.05
    , testCase "layerNorm" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[1, 2, 4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [1, 2, 4] F32) ]
                $ do
                    x <- arg @'[1, 2, 4] @'F32
                    g <- constant @'[4] @'F32 1.0
                    b <- constant @'[4] @'F32 0.0
                    y <- layerNorm x g b
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]
        bufIn <- toDeviceF32On api client dev inp [1, 2, 4]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 8
        let row1 = V.slice 4 4 result
        assertBool "layerNorm of uniform row has near-zero mean" $
            abs (V.sum row1 / 4) < 0.1
    , testCase "globalAvgPool" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[1, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [1, 4, 4, 2] F32) ]
                $ do
                    x <- arg @'[1, 4, 4, 2] @'F32
                    y <- globalAvgPool x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [if even i then 1.0 else 2.0 | i <- [0..31 :: Int]]
        bufIn <- toDeviceF32On api client dev inp [1, 4, 4, 2]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 2
        assertBool ("globalAvgPool channel 0: " ++ show (V.toList result)) $ abs (result V.! 0 - 1.0) < 0.01
        assertBool ("globalAvgPool channel 1: " ++ show (V.toList result)) $ abs (result V.! 1 - 2.0) < 0.01
    ]
