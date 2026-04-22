{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Vector.Storable as V
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..), Module)
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Device
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

main :: IO ()
main = withPJRTGPU $ \api client -> do
    putStrLn "CUDA plugin loaded."

    -- Enumerate devices
    devs <- addressableDevices api client
    putStrLn $ "Found " ++ show (length devs) ++ " addressable device(s):"
    mapM_ (\d -> do
        did  <- deviceId api d
        kind <- deviceKind api d
        putStrLn $ "  Device " ++ show did ++ " : " ++ kind
      ) devs

    -- Pick the first GPU
    mDev <- defaultGPUDevice api client
    case mDev of
        Nothing -> putStrLn "No GPU found!"
        Just dev -> do
            putStrLn "Running add on GPU..."

            let modu :: Module
                modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 2] F32)
                    , FuncArg "arg1" (TensorType [2, 2] F32)
                    ]
                    $ do
                        x <- arg
                        y <- arg
                        z <- add x y
                        return z

                mlirText = render modu

            T.putStrLn "--- MLIR ---"
            T.putStrLn mlirText
            T.putStrLn "------------"

            exec <- compile api client mlirText

            let inputA = V.fromList [1, 2, 3, 4] :: V.Vector Float
                inputB = V.fromList [10, 20, 30, 40] :: V.Vector Float

            bufA <- toDeviceOn api client dev inputA [2, 2] bufferTypeF32
            bufB <- toDeviceOn api client dev inputB [2, 2] bufferTypeF32

            [bufOut] <- executeOn api exec dev [bufA, bufB]

            result <- fromDeviceF32 api bufOut 4
            putStrLn $ "Result: " ++ show (V.toList result)
            putStrLn $ "Expected: [11.0,22.0,33.0,44.0]"

            if result == V.fromList [11, 22, 33, 44]
                then putStrLn "GPU smoke test PASSED!"
                else putStrLn "GPU smoke test FAILED!"
