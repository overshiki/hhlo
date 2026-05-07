{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module Test.Runtime.EndToEndDynamicGPU
    ( tests
    ) where

import Test.Tasty
import Test.Tasty.HUnit
import qualified Data.Vector.Storable as V

import HHLO.EDSL.Dynamic
import HHLO.IR.AST (TensorType(..))
import HHLO.Session
import HHLO.Session.Dynamic
import HHLO.Core.Types (DType(..))
import Test.Runtime.GPUResource (GPUResource(..))

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "EndToEnd.DynamicGPU"
    [ testCase "dynamic add on GPU" $ do
        GPUResource api client dev <- getGPU
        let sess = sessionFrom api client dev
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
