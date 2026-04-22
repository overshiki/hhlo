{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.MultiGPU (tests) where

import qualified Data.Vector.Storable as V
import Data.Int (Int64)
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Device
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

tests :: TestTree
tests = testGroup "Runtime.MultiGPU"
    [ testCase "execute replicas on all GPUs" executeReplicasAllGPUs
    ]

executeReplicasAllGPUs :: IO ()
executeReplicasAllGPUs = withPJRTGPU $ \api client -> do
    devs <- addressableDevices api client
    case devs of
        [] -> assertFailure "No GPU devices found"
        _  -> do
            let numDevs = length devs
            let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 2] F32)
                    , FuncArg "arg1" (TensorType [2, 2] F32)
                    ]
                    $ do
                        x <- arg @'[2, 2] @'F32
                        y <- arg @'[2, 2] @'F32
                        z <- add x y
                        return z

            exec <- compileWithOptions api client (render modu)
                        (defaultCompileOptions { optNumReplicas = numDevs })

            let inputA = V.fromList [1, 2, 3, 4] :: V.Vector Float
                inputB = V.fromList [10, 20, 30, 40] :: V.Vector Float
                dims = [2, 2] :: [Int64]

            deviceArgs <- mapM (\dev -> do
                bufA <- toDeviceOn api client dev inputA dims bufferTypeF32
                bufB <- toDeviceOn api client dev inputB dims bufferTypeF32
                return (dev, [bufA, bufB])
              ) devs

            results <- executeReplicas api exec deviceArgs

            -- Every GPU should produce the same result
            mapM_ (\(idx, outs) -> do
                let [bufOut] = outs
                result <- fromDeviceF32 api bufOut 4
                assertEqual ("GPU " ++ show idx ++ " result") result (V.fromList [11, 22, 33, 44] :: V.Vector Float)
              ) (zip [0..] results)
