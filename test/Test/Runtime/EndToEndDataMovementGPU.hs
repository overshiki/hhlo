{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndDataMovementGPU (tests) where

import Prelude hiding (map)
import qualified Data.Vector.Storable as V
import Data.Word (Word8)
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
tests getGPU = testGroup "EndToEnd.DataMovementGPU"
    [ testCase "slice 1D" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[3] @'F32 "main"
                [ FuncArg "arg0" (TensorType [5] F32) ]
                $ do
                    x <- arg @'[5] @'F32
                    y <- slice x (v1 1) (v1 4) (v1 1)
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 1.0, 2.0, 3.0, 4.0]
        bufIn <- toDeviceF32On api client dev inp [5]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 3
        result @?= V.fromList [1.0, 2.0, 3.0]
    , testCase "slice with stride" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [5] F32) ]
                $ do
                    x <- arg @'[5] @'F32
                    y <- slice x (v1 0) (v1 4) (v1 2)
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 1.0, 2.0, 3.0, 4.0]
        bufIn <- toDeviceF32On api client dev inp [5]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [0.0, 2.0]
    , testCase "pad edge" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32) ]
                $ do
                    x <- arg @'[2] @'F32
                    padVal <- constant @'[] @'F32 0.0
                    y <- pad x padVal (v1 1) (v1 1) (v1 0)
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0]
        bufIn <- toDeviceF32On api client dev inp [2]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [0.0, 1.0, 2.0, 0.0]
    , testCase "gather rows" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2, 4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3, 4] F32) ]
                $ do
                    x <- arg @'[3, 4] @'F32
                    idx <- constant @'[2] @'I64 0
                    y <- gather x idx [1] [0] [0] 1 [1, 4]
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        bufIn <- toDeviceF32On api client dev inp [3, 4]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 8
        let expected = V.fromList [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        result @?= expected
    , testCase "select true" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32)
                , FuncArg "arg1" (TensorType [2, 2] F32)
                , FuncArg "pred" (TensorType [2, 2] Bool)
                ]
                $ do
                    t <- arg @'[2, 2] @'F32
                    f <- arg @'[2, 2] @'F32
                    p <- arg @'[2, 2] @'Bool
                    y <- select p t f
                    return y
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0, 3.0, 4.0]
            b = V.fromList [5.0, 6.0, 7.0, 8.0]
            predVec = V.fromList [1, 1, 1, 1] :: V.Vector Word8
        bufA <- toDeviceF32On api client dev a [2, 2]
        bufB <- toDeviceF32On api client dev b [2, 2]
        bufP <- toDevicePredOn api client dev predVec [2, 2]
        [bufOut] <- executeOn api exec dev [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 4
        result @?= a
    , testCase "select false" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32)
                , FuncArg "arg1" (TensorType [2, 2] F32)
                , FuncArg "pred" (TensorType [2, 2] Bool)
                ]
                $ do
                    t <- arg @'[2, 2] @'F32
                    f <- arg @'[2, 2] @'F32
                    p <- arg @'[2, 2] @'Bool
                    y <- select p t f
                    return y
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0, 3.0, 4.0]
            b = V.fromList [5.0, 6.0, 7.0, 8.0]
            predVec = V.fromList [0, 0, 0, 0] :: V.Vector Word8
        bufA <- toDeviceF32On api client dev a [2, 2]
        bufB <- toDeviceF32On api client dev b [2, 2]
        bufP <- toDevicePredOn api client dev predVec [2, 2]
        [bufOut] <- executeOn api exec dev [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 4
        result @?= b
    , testCase "convert f32 to f32" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32) ]
                $ do
                    x <- arg @'[2] @'F32
                    y <- convert x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0]
        bufIn <- toDeviceF32On api client dev inp [2]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= inp
    , testCase "conditional true" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32)
                , FuncArg "arg1" (TensorType [2] F32)
                , FuncArg "pred" (TensorType [] Bool)
                ]
                $ do
                    t <- arg @'[2] @'F32
                    f <- arg @'[2] @'F32
                    p <- arg @'[] @'Bool
                    y <- conditional p (return t) (return f)
                    return y
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0]
            b = V.fromList [3.0, 4.0]
            predVec = V.fromList [1] :: V.Vector Word8
        bufA <- toDeviceF32On api client dev a [2]
        bufB <- toDeviceF32On api client dev b [2]
        bufP <- toDevicePredOn api client dev predVec []
        [bufOut] <- executeOn api exec dev [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 2
        result @?= a
    , testCase "conditional false" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32)
                , FuncArg "arg1" (TensorType [2] F32)
                , FuncArg "pred" (TensorType [] Bool)
                ]
                $ do
                    t <- arg @'[2] @'F32
                    f <- arg @'[2] @'F32
                    p <- arg @'[] @'Bool
                    y <- conditional p (return t) (return f)
                    return y
        exec <- compile api client (render modu)
        let a = V.fromList [1.0, 2.0]
            b = V.fromList [3.0, 4.0]
            predVec = V.fromList [0] :: V.Vector Word8
        bufA <- toDeviceF32On api client dev a [2]
        bufB <- toDeviceF32On api client dev b [2]
        bufP <- toDevicePredOn api client dev predVec []
        [bufOut] <- executeOn api exec dev [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 2
        result @?= b
    , testCase "map square" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[3] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3] F32) ]
                $ do
                    x <- arg @'[3] @'F32
                    y <- map [x] [0] $ \[a] -> multiply a a
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0]
        bufIn <- toDeviceF32On api client dev inp [3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 3
        result @?= V.fromList [1.0, 4.0, 9.0]
    , testCase "dynamicSlice" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [4] F32) ]
                $ do
                    x <- arg @'[4] @'F32
                    idx <- constant @'[] @'I64 1
                    y <- dynamicSlice x [idx] (v1 2)
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 1.0, 2.0, 3.0]
        bufIn <- toDeviceF32On api client dev inp [4]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [1.0, 2.0]
    , testCase "logicalAnd" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[3] @'Bool "main"
                [ FuncArg "arg0" (TensorType [3] Bool)
                , FuncArg "arg1" (TensorType [3] Bool)
                ]
                $ do
                    a <- arg @'[3] @'Bool
                    b <- arg @'[3] @'Bool
                    c <- logicalAnd a b
                    return c
        exec <- compile api client (render modu)
        let va = V.fromList [1, 1, 0] :: V.Vector Word8
            vb = V.fromList [1, 0, 0] :: V.Vector Word8
        bufA <- toDevicePredOn api client dev va [3]
        bufB <- toDevicePredOn api client dev vb [3]
        [bufOut] <- executeOn api exec dev [bufA, bufB]
        result <- fromDevice api bufOut 3 :: IO (V.Vector Word8)
        result @?= V.fromList [1, 0, 0]
    , testCase "logicalOr" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[3] @'Bool "main"
                [ FuncArg "arg0" (TensorType [3] Bool)
                , FuncArg "arg1" (TensorType [3] Bool)
                ]
                $ do
                    a <- arg @'[3] @'Bool
                    b <- arg @'[3] @'Bool
                    c <- logicalOr a b
                    return c
        exec <- compile api client (render modu)
        let va = V.fromList [1, 1, 0] :: V.Vector Word8
            vb = V.fromList [1, 0, 0] :: V.Vector Word8
        bufA <- toDevicePredOn api client dev va [3]
        bufB <- toDevicePredOn api client dev vb [3]
        [bufOut] <- executeOn api exec dev [bufA, bufB]
        result <- fromDevice api bufOut 3 :: IO (V.Vector Word8)
        result @?= V.fromList [1, 1, 0]
    , testCase "logicalNot" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[3] @'Bool "main"
                [ FuncArg "arg0" (TensorType [3] Bool) ]
                $ do
                    a <- arg @'[3] @'Bool
                    b <- logicalNot a
                    return b
        exec <- compile api client (render modu)
        let va = V.fromList [1, 0, 1] :: V.Vector Word8
        bufA <- toDevicePredOn api client dev va [3]
        [bufOut] <- executeOn api exec dev [bufA]
        result <- fromDevice api bufOut 3 :: IO (V.Vector Word8)
        result @?= V.fromList [0, 1, 0]
    , testCase "topK" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [4] F32) ]
                $ do
                    x <- arg @'[4] @'F32
                    y <- topK @'[4] @'[2] 2 0 x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [3.0, 1.0, 4.0, 1.0]
        bufIn <- toDeviceF32On api client dev inp [4]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [4.0, 3.0]
    ]
