{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndMatmulGPU (tests) where

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
tests getGPU = testGroup "EndToEnd.MatmulGPU"
    [ testCase "matmul 2D" $ do
        GPUResource api client dev <- getGPU
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
        bufA <- toDeviceF32On api client dev a [2, 3]
        bufB <- toDeviceF32On api client dev b [3, 2]
        [bufOut] <- executeOn api exec dev [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [22.0, 28.0, 49.0, 64.0]
        assertBool "matmul result close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    , testCase "linear no bias" $ do
        GPUResource api client dev <- getGPU
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
        bufIn <- toDeviceF32On api client dev inp [3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [3.0, 3.0]
    , testCase "linearBatched" $ do
        GPUResource api client dev <- getGPU
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
        bufIn <- toDeviceF32On api client dev inp [2, 3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [3.1, 3.1, 7.6, 7.6]
    , testCase "dotGeneral 3D x 2D" $ do
        GPUResource api client dev <- getGPU
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
        bufA <- toDeviceF32On api client dev a [1, 2, 3]
        bufB <- toDeviceF32On api client dev b [3, 2]
        [bufOut] <- executeOn api exec dev [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [3.0, 3.0, 7.5, 7.5]
        assertBool "dotGeneral close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    , testCase "einsum matmul" $ do
        GPUResource api client dev <- getGPU
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
        bufA <- toDeviceF32On api client dev a [2, 3]
        bufB <- toDeviceF32On api client dev b [3, 2]
        [bufOut] <- executeOn api exec dev [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [22.0, 28.0, 49.0, 64.0]
        assertBool "einsum matmul close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    , testCase "einsum transpose output" $ do
        GPUResource api client dev <- getGPU
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
        bufA <- toDeviceF32On api client dev a [2, 3]
        bufB <- toDeviceF32On api client dev b [3, 2]
        [bufOut] <- executeOn api exec dev [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [22.0, 49.0, 28.0, 64.0]
        assertBool "einsum transpose close" $
            all (\(r, e) -> abs (r - e) < 0.01) (zip (V.toList result) (V.toList expected))
    ]
