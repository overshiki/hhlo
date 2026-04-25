{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndMultiValue (tests) where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit
import Prelude hiding (compare)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..), Module)
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer
import Data.Int (Int64)

-- | A loop that counts from a starting value up to a limit, accumulating a sum.
-- Inputs: (counter_init, sum_init)
-- While counter < limit:
--   counter = counter + 1
--   sum = sum + counter
-- Returns: (final_counter, final_sum)
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

tests :: TestTree
tests = testGroup "EndToEnd.MultiValue"
    [ testCase "whileLoop2 counts and sums" $ withPJRTCPU $ \api client -> do
        let mlirText = render while2Module
        exec <- compile api client mlirText
        bufCounter <- toDevice api client (V.fromList [0 :: Int64]) [1] bufferTypeS64
        bufSum <- toDevice api client (V.fromList [0 :: Int64]) [1] bufferTypeS64
        [bufOut1, bufOut2] <- execute api exec [bufCounter, bufSum]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        -- counter goes 0->1->2->3, sum goes 0->1->3->6
        result1 @?= V.fromList [3]
        result2 @?= V.fromList [6]
    , testCase "conditional2 true branch" $ withPJRTCPU $ \api client -> do
        let mlirText = render cond2Module
        exec <- compile api client mlirText
        bufPred <- toDevice api client (V.fromList [1 :: Int64]) [1] bufferTypePred
        [bufOut1, bufOut2] <- execute api exec [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [1]
        result2 @?= V.fromList [2]
    , testCase "conditional2 false branch" $ withPJRTCPU $ \api client -> do
        let mlirText = render cond2Module
        exec <- compile api client mlirText
        bufPred <- toDevice api client (V.fromList [0 :: Int64]) [1] bufferTypePred
        [bufOut1, bufOut2] <- execute api exec [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [3]
        result2 @?= V.fromList [4]
    , testCase "whileLoop3 counts, sums and products" $ withPJRTCPU $ \api client -> do
        let mlirText = render while3Module
        exec <- compile api client mlirText
        bufCounter <- toDevice api client (V.fromList [0 :: Int64]) [1] bufferTypeS64
        bufSum <- toDevice api client (V.fromList [0 :: Int64]) [1] bufferTypeS64
        bufProd <- toDevice api client (V.fromList [1 :: Int64]) [1] bufferTypeS64
        [bufOut1, bufOut2, bufOut3] <- execute api exec [bufCounter, bufSum, bufProd]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result3 <- fromDevice api bufOut3 1 :: IO (V.Vector Int64)
        -- counter: 0->1->2->3, sum: 0->1->3->6, prod: 1->1->2->6
        result1 @?= V.fromList [3]
        result2 @?= V.fromList [6]
        result3 @?= V.fromList [6]
    , testCase "conditional3 true branch" $ withPJRTCPU $ \api client -> do
        let mlirText = render cond3Module
        exec <- compile api client mlirText
        bufPred <- toDevice api client (V.fromList [1 :: Int64]) [1] bufferTypePred
        [bufOut1, bufOut2, bufOut3] <- execute api exec [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result3 <- fromDevice api bufOut3 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [1]
        result2 @?= V.fromList [2]
        result3 @?= V.fromList [3]
    , testCase "conditional3 false branch" $ withPJRTCPU $ \api client -> do
        let mlirText = render cond3Module
        exec <- compile api client mlirText
        bufPred <- toDevice api client (V.fromList [0 :: Int64]) [1] bufferTypePred
        [bufOut1, bufOut2, bufOut3] <- execute api exec [bufPred]
        result1 <- fromDevice api bufOut1 1 :: IO (V.Vector Int64)
        result2 <- fromDevice api bufOut2 1 :: IO (V.Vector Int64)
        result3 <- fromDevice api bufOut3 1 :: IO (V.Vector Int64)
        result1 @?= V.fromList [4]
        result2 @?= V.fromList [5]
        result3 @?= V.fromList [6]
    ]
