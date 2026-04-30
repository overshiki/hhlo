{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DeriveGeneric #-}

module Test.Runtime.EndToEndAutogradGPU (tests) where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit
import GHC.Generics (Generic)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..))
import HHLO.IR.Builder (Builder, Tensor(..), arg, moduleFromBuilder, moduleFromBuilder3, tensorType)
import Data.Proxy (Proxy(..))
import HHLO.IR.Pretty
import HHLO.Autograd
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer
import HHLO.Runtime.PJRT.Types (bufferTypeF32, bufferTypePred, bufferTypeS64)
import Test.Utils
import Test.Runtime.GPUResource (GPUResource(..))

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "EndToEnd.AutogradGPU"
    [ testCase "grad sum of squares" $ do
        GPUResource api client dev <- getGPU
        let f x = do sq <- multiply x x; sumAll sq
            modu = gradModule @'[3] @'F32 f
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0]
        bufIn <- toDeviceF32On api client dev inp [3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 3
        let expected = V.fromList [2.0, 4.0, 6.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad sum of doubles" $ do
        GPUResource api client dev <- getGPU
        let f x = do d <- add x x; sumAll d
            modu = gradModule @'[3] @'F32 f
        exec <- compile api client (render modu)
        let inp = V.fromList [1.0, 2.0, 3.0]
        bufIn <- toDeviceF32On api client dev inp [3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 3
        let expected = V.fromList [2.0, 2.0, 2.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad sum of exponentials" $ do
        GPUResource api client dev <- getGPU
        let f x = do e <- exponential x; sumAll e
            modu = gradModule @'[3] @'F32 f
        exec <- compile api client (render modu)
        let inp = V.fromList [0.0, 0.0, 0.0]
        bufIn <- toDeviceF32On api client dev inp [3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 3
        let expected = V.fromList [1.0, 1.0, 1.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad matmul" $ do
        GPUResource api client dev <- getGPU
        let f x = do
                w <- constant @'[3, 2] @'F32 0.5
                y <- matmul x w
                sumAll y
            gradModu = gradModule @'[2, 3] @'F32 f
        exec <- compile api client (render gradModu)
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        bufIn <- toDeviceF32On api client dev inp [2, 3]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 6
        let expected = V.fromList [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assertBool "grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad avgPool" $ do
        GPUResource api client dev <- getGPU
        let f x = do
                let windowDims = v4 1 2 2 1
                    strides    = v4 1 2 2 1
                    padding    = v4 (0, 0) (0, 0) (0, 0) (0, 0)
                initVal <- constant @'[] @'F32 0.0
                y <- reduceWindow windowDims strides padding "stablehlo.add" initVal x
                divisor <- constant @'[] @'F32 4.0
                divisorBC <- broadcastWithDims @'[] @'[1, 2, 2, 1] [] divisor
                z <- divide y divisorBC
                sumAll z
            gradModu = gradModule @'[1, 4, 4, 1] @'F32 f
        exec <- compile api client (render gradModu)
        let inp = V.fromList [1.0..16.0]
        bufIn <- toDeviceF32On api client dev inp [1, 4, 4, 1]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 16
        let expected = V.fromList (replicate 16 0.25)
        assertBool "avgPool grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad conv2d" $ do
        GPUResource api client dev <- getGPU
        let f x = do
                k <- constant @'[2, 2, 1, 1] @'F32 1.0
                y <- conv2d @1 @3 @3 @1 @1 @2 @2 @2 @2 x k
                sumAll y
            gradModu = gradModule @'[1, 3, 3, 1] @'F32 f
        exec <- compile api client (render gradModu)
        let inp = V.fromList [1.0..9.0]
        bufIn <- toDeviceF32On api client dev inp [1, 3, 3, 1]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 9
        let expected = V.fromList [1, 2, 1, 2, 4, 2, 1, 2, 1]
        assertBool "conv2d grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad maxPool" $ do
        GPUResource api client dev <- getGPU
        let f x = do
                let kernel = v2 2 2
                    stride = v2 2 2
                    padding = p2 (0,0) (0,0)
                y <- maxPool @1 @4 @4 @1 @2 @2 kernel stride padding x
                sumAll y
            gradModu = gradModule @'[1, 4, 4, 1] @'F32 f
        exec <- compile api client (render gradModu)
        let inp = V.fromList [1.0..16.0]
        bufIn <- toDeviceF32On api client dev inp [1, 4, 4, 1]
        [bufOut] <- executeOn api exec dev [bufIn]
        result <- fromDeviceF32 api bufOut 16
        let expected = V.fromList [0,0,0,0, 0,1,0,1, 0,0,0,0, 0,1,0,1]
        assertBool "maxPool grad close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "grad concatenate2" $ do
        GPUResource api client dev <- getGPU
        let f a b = do
                c <- concatenate2 @'[2, 4] @'[2, 4] @'[2, 8] @'F32 1 a b
                sumAll c
            modu = moduleFromBuilder @'[4, 4] @'F32 "main"
                [ FuncArg "arg0" (tensorType (Proxy @'[2, 4]) (Proxy @'F32))
                , FuncArg "arg1" (tensorType (Proxy @'[2, 4]) (Proxy @'F32))
                ] $ do
                    a <- arg @'[2, 4] @'F32
                    b <- arg @'[2, 4] @'F32
                    (da, db) <- grad2 f a b
                    concatenate 0 [da, db]
        exec <- compile api client (render modu)
        let inp1 = V.fromList [1..8]
            inp2 = V.fromList [1..8]
        bufIn1 <- toDeviceF32On api client dev inp1 [2, 4]
        bufIn2 <- toDeviceF32On api client dev inp2 [2, 4]
        [bufOut] <- executeOn api exec dev [bufIn1, bufIn2]
        result <- fromDeviceF32 api bufOut 16
        let expected = V.fromList (replicate 16 1.0)
        result @?= expected
    , testCase "grad concatenate3" $ do
        GPUResource api client dev <- getGPU
        let f a b c = do
                x <- concatenate @'[2, 4] @'[2, 12] @'F32 1 [a, b, c]
                sumAll x
            modu = moduleFromBuilder @'[6, 4] @'F32 "main"
                [ FuncArg "arg0" (tensorType (Proxy @'[2, 4]) (Proxy @'F32))
                , FuncArg "arg1" (tensorType (Proxy @'[2, 4]) (Proxy @'F32))
                , FuncArg "arg2" (tensorType (Proxy @'[2, 4]) (Proxy @'F32))
                ] $ do
                    a <- arg @'[2, 4] @'F32
                    b <- arg @'[2, 4] @'F32
                    c <- arg @'[2, 4] @'F32
                    (da, db, dc) <- grad3 f a b c
                    concatenate 0 [da, db, dc]
        exec <- compile api client (render modu)
        let inp = V.fromList [1..8]
        bufIn1 <- toDeviceF32On api client dev inp [2, 4]
        bufIn2 <- toDeviceF32On api client dev inp [2, 4]
        bufIn3 <- toDeviceF32On api client dev inp [2, 4]
        [bufOut] <- executeOn api exec dev [bufIn1, bufIn2, bufIn3]
        result <- fromDeviceF32 api bufOut 24
        let expected = V.fromList (replicate 24 1.0)
        result @?= expected
    , testCase "grad2 multiply" $ do
        GPUResource api client dev <- getGPU
        let f x y = do z <- multiply x y; sumAll z
            modu = moduleFromBuilder @'[4] @'F32 "main"
                [ FuncArg "arg0" (tensorType (Proxy @'[2]) (Proxy @'F32))
                , FuncArg "arg1" (tensorType (Proxy @'[2]) (Proxy @'F32))
                ] $ do
                    x <- arg @'[2] @'F32
                    y <- arg @'[2] @'F32
                    (dx, dy) <- grad2 f x y
                    concatenate 0 [dx, dy]
        exec <- compile api client (render modu)
        let inp1 = V.fromList [1.0, 2.0]
            inp2 = V.fromList [3.0, 4.0]
        bufIn1 <- toDeviceF32On api client dev inp1 [2]
        bufIn2 <- toDeviceF32On api client dev inp2 [2]
        [bufOut] <- executeOn api exec dev [bufIn1, bufIn2]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [3.0, 4.0, 1.0, 2.0]
        assertBool "grad2 close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "gradWithParams nested" $ do
        GPUResource api client dev <- getGPU
        let loss :: ModelParams -> Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
            loss p x = do
                y1a <- multiply x (lw (layer1 p))
                y1 <- add y1a (lb (layer1 p))
                y2a <- multiply x (lw (layer2 p))
                y2 <- add y2a (lb (layer2 p))
                ysum <- add y1 y2
                sumAll ysum
            modu = moduleFromBuilder @'[8] @'F32 "main"
                [ FuncArg "arg0" (tensorType (Proxy @'[2]) (Proxy @'F32))
                , FuncArg "arg1" (tensorType (Proxy @'[2]) (Proxy @'F32))
                , FuncArg "arg2" (tensorType (Proxy @'[2]) (Proxy @'F32))
                , FuncArg "arg3" (tensorType (Proxy @'[2]) (Proxy @'F32))
                , FuncArg "arg4" (tensorType (Proxy @'[2]) (Proxy @'F32))
                ] $ do
                    l1w <- arg @'[2] @'F32
                    l1b <- arg @'[2] @'F32
                    l2w <- arg @'[2] @'F32
                    l2b <- arg @'[2] @'F32
                    xIn <- arg @'[2] @'F32
                    let params = ModelParams (LayerParams l1w l1b) (LayerParams l2w l2b)
                    grads <- gradWithParams loss params xIn
                    packed <- paramPack grads
                    return (btoTyped @'[8] @'F32 packed)
        exec <- compile api client (render modu)
        let l1wVal = V.fromList [1.0, 1.0]
            l1bVal = V.fromList [0.0, 0.0]
            l2wVal = V.fromList [1.0, 1.0]
            l2bVal = V.fromList [0.0, 0.0]
            xVal   = V.fromList [3.0, 4.0]
        bufL1W <- toDeviceF32On api client dev l1wVal [2]
        bufL1B <- toDeviceF32On api client dev l1bVal [2]
        bufL2W <- toDeviceF32On api client dev l2wVal [2]
        bufL2B <- toDeviceF32On api client dev l2bVal [2]
        bufX   <- toDeviceF32On api client dev xVal [2]
        [bufOut] <- executeOn api exec dev [bufL1W, bufL1B, bufL2W, bufL2B, bufX]
        result <- fromDeviceF32 api bufOut 8
        let expected = V.fromList [3.0, 4.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0]
        assertBool "gradWithParams nested close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    , testCase "gradWithParams" $ do
        GPUResource api client dev <- getGPU
        let loss :: MLPParams -> Tensor '[2] 'F32 -> Builder (Tensor '[] 'F32)
            loss p x = do
                y1 <- multiply x (w p)
                y <- add y1 (b p)
                sumAll y
            modu = moduleFromBuilder @'[4] @'F32 "main"
                [ FuncArg "arg0" (tensorType (Proxy @'[2]) (Proxy @'F32))
                , FuncArg "arg1" (tensorType (Proxy @'[2]) (Proxy @'F32))
                , FuncArg "arg2" (tensorType (Proxy @'[2]) (Proxy @'F32))
                ] $ do
                    wIn <- arg @'[2] @'F32
                    bIn <- arg @'[2] @'F32
                    xIn <- arg @'[2] @'F32
                    let params = MLPParams wIn bIn
                    grads <- gradWithParams loss params xIn
                    packed <- paramPack grads
                    return (btoTyped @'[4] @'F32 packed)
        exec <- compile api client (render modu)
        let wVal = V.fromList [1.0, 2.0]
            bVal = V.fromList [0.0, 0.0]
            xVal = V.fromList [3.0, 4.0]
        bufW <- toDeviceF32On api client dev wVal [2]
        bufB <- toDeviceF32On api client dev bVal [2]
        bufX <- toDeviceF32On api client dev xVal [2]
        [bufOut] <- executeOn api exec dev [bufW, bufB, bufX]
        result <- fromDeviceF32 api bufOut 4
        let expected = V.fromList [3.0, 4.0, 1.0, 1.0]
        assertBool "gradWithParams close" $
            V.and (V.zipWith (\r e -> abs (r - e) < 0.01) result expected)
    ]

data LayerParams = LayerParams
    { lw :: Tensor '[2] 'F32
    , lb :: Tensor '[2] 'F32
    } deriving (Generic)

instance ParamTree LayerParams

data ModelParams = ModelParams
    { layer1 :: LayerParams
    , layer2 :: LayerParams
    } deriving (Generic)

instance ParamTree ModelParams

data MLPParams = MLPParams
    { w :: Tensor '[2] 'F32
    , b :: Tensor '[2] 'F32
    } deriving (Generic)

instance ParamTree MLPParams
