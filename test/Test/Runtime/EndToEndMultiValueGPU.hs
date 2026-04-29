{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndMultiValueGPU (tests) where

import qualified Data.Vector.Storable as V
import Data.Int (Int64)
import Data.Word (Word8)
import Test.Tasty
import Test.Tasty.HUnit
import Prelude hiding (compare)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..), Module)
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer
import HHLO.Runtime.PJRT.Types (bufferTypeF32, bufferTypePred, bufferTypeS64)
import HHLO.Runtime.PJRT.Types (bufferTypeS64, bufferTypePred)
import Test.Utils
import Test.Runtime.GPUResource (GPUResource(..))

while2Module :: Module
while2Module =
    moduleFromBuilder2 @'[] @'I64 @'[] @'I64 "main"
        [ FuncArg "arg0" (TensorType [] I64)
        , FuncArg "arg1" (TensorType [] I64)
        ]
        $ do
            counter0 <- arg @'[] @'I64
            sum0 <- arg @'[] @'I64
            result <- whileLoop2 counter0 sum0
                (\c _s -> do
                    limitC <- constant @'[] @'I64 3
                    cond <- compare c limitC "LT"
                    return cond)
                (\c s -> do
                    one <- constant @'[] @'I64 1
                    cNext <- add c one
                    sNext <- add s cNext
                    returnTuple2 cNext sNext)
            return result

cond2Module :: Module
cond2Module =
    moduleFromBuilder2 @'[] @'I64 @'[] @'I64 "main"
        [ FuncArg "arg0" (TensorType [] Bool) ]
        $ do
            p <- arg @'[] @'Bool
            t1 <- constant @'[] @'I64 1
            t2 <- constant @'[] @'I64 2
            f1 <- constant @'[] @'I64 3
            f2 <- constant @'[] @'I64 4
            result <- conditional2 p (returnTuple2 t1 t2) (returnTuple2 f1 f2)
            return result

while3Module :: Module
while3Module =
    moduleFromBuilder3 @'[] @'I64 @'[] @'I64 @'[] @'I64 "main"
        [ FuncArg "arg0" (TensorType [] I64)
        , FuncArg "arg1" (TensorType [] I64)
        , FuncArg "arg2" (TensorType [] I64)
        ]
        $ do
            counter0 <- arg @'[] @'I64
            sum0 <- arg @'[] @'I64
            prod0 <- arg @'[] @'I64
            result <- whileLoop3 counter0 sum0 prod0
                (\c _s _p -> do
                    limitC <- constant @'[] @'I64 3
                    cond <- compare c limitC "LT"
                    return cond)
                (\c s p -> do
                    one <- constant @'[] @'I64 1
                    cNext <- add c one
                    sNext <- add s cNext
                    pNext <- multiply p cNext
                    returnTuple3 cNext sNext pNext)
            return result

cond3Module :: Module
cond3Module =
    moduleFromBuilder3 @'[] @'I64 @'[] @'I64 @'[] @'I64 "main"
        [ FuncArg "arg0" (TensorType [] Bool) ]
        $ do
            p <- arg @'[] @'Bool
            t1 <- constant @'[] @'I64 1
            t2 <- constant @'[] @'I64 2
            t3 <- constant @'[] @'I64 3
            f1 <- constant @'[] @'I64 4
            f2 <- constant @'[] @'I64 5
            f3 <- constant @'[] @'I64 6
            result <- conditional3 p (returnTuple3 t1 t2 t3) (returnTuple3 f1 f2 f3)
            return result

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "EndToEnd.MultiValueGPU"
    [ testCase "whileLoop2 counts and sums" $ do
        GPUResource api client dev <- getGPU
        let mlirText = render while2Module
        exec <- compile api client mlirText
        bufCounter <- toDeviceS64On api client dev (V.fromList [0 :: Int64]) [1]
        bufSum <- toDeviceS64On api client dev (V.fromList [0 :: Int64]) [1]
        [bufOut1, bufOut2] <- executeOn api exec dev [bufCounter, bufSum]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [3]
        result2 @?= V.fromList [6]
    , testCase "conditional2 true branch" $ do
        GPUResource api client dev <- getGPU
        let mlirText = render cond2Module
        exec <- compile api client mlirText
        bufPred <- toDevicePredOn api client dev (V.fromList [1 :: Word8]) [1]
        [bufOut1, bufOut2] <- executeOn api exec dev [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [1]
        result2 @?= V.fromList [2]
    , testCase "conditional2 false branch" $ do
        GPUResource api client dev <- getGPU
        let mlirText = render cond2Module
        exec <- compile api client mlirText
        bufPred <- toDevicePredOn api client dev (V.fromList [0 :: Word8]) [1]
        [bufOut1, bufOut2] <- executeOn api exec dev [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [3]
        result2 @?= V.fromList [4]
    , testCase "whileLoop3 counts, sums and products" $ do
        GPUResource api client dev <- getGPU
        let mlirText = render while3Module
        exec <- compile api client mlirText
        bufCounter <- toDeviceS64On api client dev (V.fromList [0 :: Int64]) [1]
        bufSum <- toDeviceS64On api client dev (V.fromList [0 :: Int64]) [1]
        bufProd <- toDeviceS64On api client dev (V.fromList [1 :: Int64]) [1]
        [bufOut1, bufOut2, bufOut3] <- executeOn api exec dev [bufCounter, bufSum, bufProd]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result3 <- fromDevice api bufOut3 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [3]
        result2 @?= V.fromList [6]
        result3 @?= V.fromList [6]
    , testCase "conditional3 true branch" $ do
        GPUResource api client dev <- getGPU
        let mlirText = render cond3Module
        exec <- compile api client mlirText
        bufPred <- toDevicePredOn api client dev (V.fromList [1 :: Word8]) [1]
        [bufOut1, bufOut2, bufOut3] <- executeOn api exec dev [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result3 <- fromDevice api bufOut3 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [1]
        result2 @?= V.fromList [2]
        result3 @?= V.fromList [3]
    , testCase "conditional3 false branch" $ do
        GPUResource api client dev <- getGPU
        let mlirText = render cond3Module
        exec <- compile api client mlirText
        bufPred <- toDevicePredOn api client dev (V.fromList [0 :: Word8]) [1]
        [bufOut1, bufOut2, bufOut3] <- executeOn api exec dev [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result3 <- fromDevice api bufOut3 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [4]
        result2 @?= V.fromList [5]
        result3 @?= V.fromList [6]
    ]
