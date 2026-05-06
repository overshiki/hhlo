{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module Test.Runtime.EndToEndDynamic
    ( tests
    ) where

import Test.Tasty
import Test.Tasty.HUnit
import qualified Data.Vector.Storable as V

import HHLO.EDSL.Dynamic
import HHLO.IR.AST (TensorType(..), FuncArg(..))
import HHLO.IR.Builder (moduleFromBuilderDynamic)
import HHLO.Session
import HHLO.Session.Dynamic
import HHLO.Core.Types (DType(..))

tests :: TestTree
tests = testGroup "EndToEnd.Dynamic"
    [ testCase "dynamic add on CPU" $ withCPU $ \sess -> do
        let dm = dynamicModule "main"
                [ TensorType [Nothing] F32 ]
                $ \argTypes -> do
                    a <- anyArg (head argTypes)
                    b <- anyAdd a a
                    return (anyVid b, anyType b)
        compiled <- compileDynamic sess dm

        let input3 = dynamicHostFromVector @'F32 (V.fromList [1.0, 2.0, 3.0]) [3]
        [out3] <- runDynamicCompiled compiled [input3]
        dhtShape out3 @?= [3]
        dhtData out3 @?= V.fromList [2.0, 4.0, 6.0]

        let input5 = dynamicHostFromVector @'F32 (V.fromList [1.0, 2.0, 3.0, 4.0, 5.0]) [5]
        [out5] <- runDynamicCompiled compiled [input5]
        dhtShape out5 @?= [5]
        dhtData out5 @?= V.fromList [2.0, 4.0, 6.0, 8.0, 10.0]

    , testCase "dynamic add on GPU" $ withGPU $ \sess -> do
        let dm = dynamicModule "main"
                [ TensorType [Nothing] F32 ]
                $ \argTypes -> do
                    a <- anyArg (head argTypes)
                    b <- anyAdd a a
                    return (anyVid b, anyType b)
        compiled <- compileDynamic sess dm

        let input4 = dynamicHostFromVector @'F32 (V.fromList [10.0, 20.0, 30.0, 40.0]) [4]
        [out4] <- runDynamicCompiled compiled [input4]
        dhtShape out4 @?= [4]
        dhtData out4 @?= V.fromList [20.0, 40.0, 60.0, 80.0]

    ]
