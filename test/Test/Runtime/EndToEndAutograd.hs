{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndAutograd where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Pretty
import HHLO.Autograd
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer
import Test.Utils

tests :: TestTree
tests = testGroup "EndToEnd.Autograd"
    [ testCase "grad sum of squares" $ withPJRTCPU $ \api client -> do
        let f x = do sq <- multiply x x; sumAll sq
            modu = gradModule @'[3] @'F32 f
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0]
        bufIn <- toDeviceF32 api client inp [3]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 3
        -- grad = 2 * x = [2, 4, 6]
        let expected = V.fromList [2.0, 4.0, 6.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad sum of doubles" $ withPJRTCPU $ \api client -> do
        let f x = do d <- add x x; sumAll d
            modu = gradModule @'[3] @'F32 f
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0]
        bufIn <- toDeviceF32 api client inp [3]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 3
        -- grad = [2, 2, 2]
        let expected = V.fromList [2.0, 2.0, 2.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad sum of exponentials" $ withPJRTCPU $ \api client -> do
        let f x = do e <- exponential x; sumAll e
            modu = gradModule @'[3] @'F32 f
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 0.0, 0.0]
        bufIn <- toDeviceF32 api client inp [3]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 3
        -- grad = exp(x) = [1, 1, 1]
        let expected = V.fromList [1.0, 1.0, 1.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad matmul" $ withPJRTCPU $ \api client -> do
        let f x = do
                w <- constant @'[3, 2] @'F32 0.5
                y <- matmul x w
                sumAll y
            gradModu = gradModule @'[2, 3] @'F32 f
        exec <- compile api client (render gradModu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        bufIn <- toDeviceF32 api client inp [2, 3]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 6
        -- grad = sum over cols of W = [0.5+0.5, 0.5+0.5, 0.5+0.5] for each row
        -- = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        let expected = V.fromList [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    ]
