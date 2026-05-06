{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndReductionsGPU (tests) where

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
tests getGPU = testGroup "EndToEnd.ReductionsGPU"
    [ testCase "reduceSum all" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 2, Just 3] F32) ]
                $ do
                    x <- arg @'[2, 3] @'F32
                    y <- reduceSum x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        bufIn <- toDeviceF32On api client dev inp [2, 3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 1
        result @?= V.fromList [21.0]
    , testCase "maxPool 2x2" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 1, Just 4, Just 4, Just 1] F32) ]
                $ do
                    x <- arg @'[1, 4, 4, 1] @'F32
                    y <- maxPool (v2 2 2) (v2 2 2) (p2 (0,0) (0,0)) x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                              9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        bufIn <- toDeviceF32On api client dev inp [1, 4, 4, 1]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 4
        result @?= V.fromList [6.0, 8.0, 14.0, 16.0]
    , testCase "avgPool 2x2" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 1, Just 4, Just 4, Just 1] F32) ]
                $ do
                    x <- arg @'[1, 4, 4, 1] @'F32
                    y <- avgPool (v2 2 2) (v2 2 2) x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                              9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        bufIn <- toDeviceF32On api client dev inp [1, 4, 4, 1]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [3.5, 5.5, 11.5, 13.5]
        assertBool "avgPool close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "productAll" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 2, Just 3] F32) ]
                $ do
                    x <- arg @'[2, 3] @'F32
                    y <- productAll x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        bufIn <- toDeviceF32On api client dev inp [2, 3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 1
        result @?= V.fromList [720.0]
    , testCase "productDim" $ do
        GPUResource api client dev <- getGPU
        let modu = moduleFromBuilder @'[2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 2, Just 3] F32) ]
                $ do
                    x <- arg @'[2, 3] @'F32
                    y <- productDim @'[2, 3] @'[2] [1] x
                    return y
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        bufIn <- toDeviceF32On api client dev inp [2, 3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 2
        result @?= V.fromList [6.0, 120.0]
    ]
