{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndDataMovement where

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
import HHLO.Runtime.PJRT.Types (bufferTypePred)
import Test.Utils

tests :: TestTree
tests = testGroup "EndToEnd.DataMovement"
    [ testCase "slice 1D" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[3] @'F32 "main"
                [ FuncArg "arg0" (TensorType [5] F32) ]
                $ do
                    x <- arg @'[5] @'F32
                    y <- slice x [1] [4] [1]
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 1.0, 2.0, 3.0, 4.0]
        bufIn <- toDeviceF32 api client inp [5]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 3
        result @?= V.fromList [1.0, 2.0, 3.0]
    , testCase "slice with stride" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [5] F32) ]
                $ do
                    x <- arg @'[5] @'F32
                    y <- slice x [0] [4] [2]
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 1.0, 2.0, 3.0, 4.0]
        bufIn <- toDeviceF32 api client inp [5]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [0.0, 2.0]
    , testCase "pad edge" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32) ]
                $ do
                    x <- arg @'[2] @'F32
                    padVal <- constant @'[] @'F32 0.0
                    y <- pad x padVal [1] [1] [0]
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0]
        bufIn <- toDeviceF32 api client inp [2]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 4
        -- pad with 0.0: [0, 1, 2, 0]
        result @?= V.fromList [0.0, 1.0, 2.0, 0.0]
    , testCase "gather rows" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3, 4] F32) ]
                $ do
                    x <- arg @'[3, 4] @'F32
                    idx <- constant @'[2] @'I64 0
                    y <- gather x idx [1] [0] [0] 1 [1, 4]
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        bufIn <- toDeviceF32 api client inp [3, 4]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 8
        -- gather rows 0 and 0 (since idx is [0,0])
        let expected = V.fromList [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        result @?= expected
    , testCase "select true" $ withPJRTCPU $ \api client -> do
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
            pred = V.fromList [1, 1, 1, 1] :: V.Vector Word8  -- i1 = Word8
        bufA <- toDeviceF32 api client a [2, 2]
        bufB <- toDeviceF32 api client b [2, 2]
        bufP <- toDevice api client pred [2, 2] bufferTypePred
        [bufOut] <- execute api exec [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 4
        result @?= a
    , testCase "select false" $ withPJRTCPU $ \api client -> do
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
            pred = V.fromList [0, 0, 0, 0] :: V.Vector Word8
        bufA <- toDeviceF32 api client a [2, 2]
        bufB <- toDeviceF32 api client b [2, 2]
        bufP <- toDevice api client pred [2, 2] bufferTypePred
        [bufOut] <- execute api exec [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 4
        result @?= b
    , testCase "convert f32 to f32" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2] F32) ]
                $ do
                    x <- arg @'[2] @'F32
                    y <- convert x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0]
        bufIn <- toDeviceF32 api client inp [2]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= inp
    , testCase "conditional true" $ withPJRTCPU $ \api client -> do
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
            pred = V.fromList [1] :: V.Vector Word8
        bufA <- toDeviceF32 api client a [2]
        bufB <- toDeviceF32 api client b [2]
        bufP <- toDevice api client pred [] bufferTypePred
        [bufOut] <- execute api exec [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 2
        result @?= a
    , testCase "conditional false" $ withPJRTCPU $ \api client -> do
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
            pred = V.fromList [0] :: V.Vector Word8
        bufA <- toDeviceF32 api client a [2]
        bufB <- toDeviceF32 api client b [2]
        bufP <- toDevice api client pred [] bufferTypePred
        [bufOut] <- execute api exec [bufA, bufB, bufP]
        result <- fromDeviceF32 api bufOut 2
        result @?= b
    , testCase "map square" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[3] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3] F32) ]
                $ do
                    x <- arg @'[3] @'F32
                    y <- map [x] [0] $ \[a] -> multiply a a
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0]
        bufIn <- toDeviceF32 api client inp [3]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 3
        result @?= V.fromList [1.0, 4.0, 9.0]
    , testCase "dynamicSlice" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [4] F32) ]
                $ do
                    x <- arg @'[4] @'F32
                    idx <- constant @'[] @'I64 1
                    y <- dynamicSlice x [idx] [2]
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 1.0, 2.0, 3.0]
        bufIn <- toDeviceF32 api client inp [4]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [1.0, 2.0]
    , testCase "logicalAnd" $ withPJRTCPU $ \api client -> do
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
        bufA <- toDevice api client va [3] bufferTypePred
        bufB <- toDevice api client vb [3] bufferTypePred
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDevice api bufOut 3 :: IO (V.Vector Word8)
        result @?= V.fromList [1, 0, 0]
    , testCase "logicalOr" $ withPJRTCPU $ \api client -> do
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
        bufA <- toDevice api client va [3] bufferTypePred
        bufB <- toDevice api client vb [3] bufferTypePred
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDevice api bufOut 3 :: IO (V.Vector Word8)
        result @?= V.fromList [1, 1, 0]
    , testCase "logicalNot" $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[3] @'Bool "main"
                [ FuncArg "arg0" (TensorType [3] Bool) ]
                $ do
                    a <- arg @'[3] @'Bool
                    b <- logicalNot a
                    return b
        exec <- compile api client (render modu)
        let va = V.fromList [1, 0, 1] :: V.Vector Word8
        bufA <- toDevice api client va [3] bufferTypePred
        [bufOut] <- execute api exec [bufA]
        result <- fromDevice api bufOut 3 :: IO (V.Vector Word8)
        result @?= V.fromList [0, 1, 0]
    ]
