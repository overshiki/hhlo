{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Dynamic shapes example: compile once per unique shape, run everywhere.
--
-- This uses 'HHLO.Session.Dynamic' which automatically specialises the
-- module to concrete shapes at runtime.  It works on both CPU and GPU.
--
-- Build with:
--   cabal build example-dynamic-shapes -fexamples
-- Run with:
--   cabal run example-dynamic-shapes -fexamples
--
module Main where

import qualified Data.Vector.Storable as V

import HHLO.EDSL.Dynamic
import HHLO.IR.AST (TensorType(..))
import HHLO.Session
import HHLO.Session.Dynamic
import HHLO.Core.Types (DType(..))

-- | Build a dynamic module template: input -> input + input.
dynamicAddModule :: DynamicModule
dynamicAddModule = dynamicModule "main"
    [ TensorType [Nothing] F32 ]
    $ \argTypes -> do
        a <- anyArg (head argTypes)
        b <- anyAdd a a
        return (anyVid b, anyType b)

main :: IO ()
main = do
    putStrLn "=== Dynamic Shapes Example ==="

    -- Run on CPU
    putStrLn "\n--- CPU ---"
    withCPU $ \sess -> do
        compiled <- compileDynamic sess dynamicAddModule

        -- Run with shape [3]
        let input3 = dynamicHostFromVector @'F32 (V.fromList [1.0, 2.0, 3.0]) [3]
        [out3] <- runDynamicCompiled compiled [input3]
        putStrLn $ "Input [3]:  [1,2,3] * 2 = " ++ show (V.toList (dhtData out3))
        putStrLn $ "Output shape: " ++ show (dhtShape out3)

        -- Run with shape [5] — automatically compiles a new static module
        let input5 = dynamicHostFromVector @'F32 (V.fromList [1.0, 2.0, 3.0, 4.0, 5.0]) [5]
        [out5] <- runDynamicCompiled compiled [input5]
        putStrLn $ "Input [5]:  [1,2,3,4,5] * 2 = " ++ show (V.toList (dhtData out5))
        putStrLn $ "Output shape: " ++ show (dhtShape out5)

        -- Run with shape [3] again — reuses the cached compiled module
        [out3b] <- runDynamicCompiled compiled [input3]
        putStrLn $ "Input [3] again: [1,2,3] * 2 = " ++ show (V.toList (dhtData out3b))

    -- Run on GPU
    putStrLn "\n--- GPU ---"
    withGPU $ \sess -> do
        compiled <- compileDynamic sess dynamicAddModule

        let input4 = dynamicHostFromVector @'F32 (V.fromList [10.0, 20.0, 30.0, 40.0]) [4]
        [out4] <- runDynamicCompiled compiled [input4]
        putStrLn $ "Input [4]:  [10,20,30,40] * 2 = " ++ show (V.toList (dhtData out4))
        putStrLn $ "Output shape: " ++ show (dhtShape out4)

        let input2 = dynamicHostFromVector @'F32 (V.fromList [5.0, 7.0]) [2]
        [out2] <- runDynamicCompiled compiled [input2]
        putStrLn $ "Input [2]:  [5,7] * 2 = " ++ show (V.toList (dhtData out2))
        putStrLn $ "Output shape: " ++ show (dhtShape out2)

    putStrLn "\n=== Done ==="
