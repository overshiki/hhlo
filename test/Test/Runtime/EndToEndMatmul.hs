{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndMatmul where

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

tests :: TestTree
tests = testGroup "EndToEnd.Matmul"
    [ testCase "matmul 2D" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 3] F32)
                , FuncArg "arg1" (TensorType [3, 2] F32)
                ]
                $ do
                    x <- arg @'[2, 3] @'F32
                    y <- arg @'[3, 2] @'F32
                    z <- matmul x y
                    return z
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: V.Vector Float
            b = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: V.Vector Float
        bufA <- toDeviceF32 api client a [2, 3]
        bufB <- toDeviceF32 api client b [3, 2]
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        -- [1*1+2*3+3*5, 1*2+2*4+3*6, 4*1+5*3+6*5, 4*2+5*4+6*6]
        -- = [22, 28, 49, 64]
        let expected = V.fromList [22.0, 28.0, 49.0, 64.0]
        assertBool "matmul result close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    , testCase "linear no bias" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3] F32) ]
                $ do
                    x <- arg @'[3] @'F32
                    w <- constant @'[3, 2] @'F32 0.5
                    b <- constant @'[2] @'F32 0.0
                    z <- linear x w b
                    return z
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0]
        bufIn <- toDeviceF32 api client inp [3]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 2
        -- [1*0.5+2*0.5+3*0.5, 1*0.5+2*0.5+3*0.5] = [3, 3]
        result @?= V.fromList [3.0, 3.0]
    , testCase "linearBatched" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 3] F32) ]
                $ do
                    x <- arg
                    w <- constant @'[3, 2] @'F32 0.5
                    b <- constant @'[2] @'F32 0.1
                    z <- linearBatched x w b
                    return z
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        bufIn <- toDeviceF32 api client inp [2, 3]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 4
        -- Row 0: [3.0+0.1, 3.0+0.1] = [3.1, 3.1]
        -- Row 1: [7.5+0.1, 7.5+0.1] = [7.6, 7.6]
        result @?= V.fromList [3.1, 3.1, 7.6, 7.6]
    , testCase "dotGeneral 3D x 2D" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[1, 2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [1, 2, 3] F32)
                , FuncArg "arg1" (TensorType [3, 2] F32)
                ]
                $ do
                    x <- arg @'[1, 2, 3] @'F32
                    y <- arg @'[3, 2] @'F32
                    z <- dotGeneral @'[1, 2, 3] @'[3, 2] @'[1, 2, 2] @'F32 [] [] [2] [0] x y
                    return z
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            b = V.fromList [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        bufA <- toDeviceF32 api client a [1, 2, 3]
        bufB <- toDeviceF32 api client b [3, 2]
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        -- row0: (1+2+3)*0.5 = 3.0, row1: (4+5+6)*0.5 = 7.5
        let expected = V.fromList [3.0, 3.0, 7.5, 7.5]
        assertBool "dotGeneral close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    , testCase "einsum matmul" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 3] F32)
                , FuncArg "arg1" (TensorType [3, 2] F32)
                ]
                $ do
                    x <- arg @'[2, 3] @'F32
                    y <- arg @'[3, 2] @'F32
                    z <- einsum "ij,jk->ik" x y
                    return z
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: V.Vector Float
            b = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: V.Vector Float
        bufA <- toDeviceF32 api client a [2, 3]
        bufB <- toDeviceF32 api client b [3, 2]
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [22.0, 28.0, 49.0, 64.0]
        assertBool "einsum matmul close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    , testCase "einsum transpose output" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 3] F32)
                , FuncArg "arg1" (TensorType [3, 2] F32)
                ]
                $ do
                    x <- arg @'[2, 3] @'F32
                    y <- arg @'[3, 2] @'F32
                    z <- einsum "ij,jk->ki" x y
                    return z
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: V.Vector Float
            b = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: V.Vector Float
        bufA <- toDeviceF32 api client a [2, 3]
        bufB <- toDeviceF32 api client b [3, 2]
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        -- Same as matmul but transposed: [[22,49],[28,64]] flattened row-major = [22,49,28,64]
        let expected = V.fromList [22.0, 49.0, 28.0, 64.0]
        assertBool "einsum transpose close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    ]
