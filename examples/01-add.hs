{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 1: Element-wise addition of two 2x2 matrices.
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run hhlo-demo
-- Or compile this file directly:
--   ghc -isrc -Icbits examples/01-add.hs cbits/pjrt_shim.c -ldl -lstdc++

module Main where

import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

main :: IO ()
main = do
    putStrLn "=== Example 1: Element-wise Addition ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- Build program using EDSL
    let modu = moduleFromBuilder @'[2,2] @'F32 "main"
            [ FuncArg "a" (TensorType [2, 2] F32)
            , FuncArg "b" (TensorType [2, 2] F32)
            ]
            $ do
                a <- arg
                b <- arg
                c <- add a b
                return c

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    -- Compile and run
    exec <- compile api client (render modu)

    let inputA = V.fromList [1, 2, 3, 4] :: V.Vector Float
        inputB = V.fromList [10, 20, 30, 40] :: V.Vector Float
    bufA <- toDeviceF32 api client inputA [2, 2]
    bufB <- toDeviceF32 api client inputB [2, 2]

    [bufC] <- execute api exec [bufA, bufB]
    result <- fromDeviceF32 api bufC 4

    putStrLn $ "Input A:  " ++ show (V.toList inputA)
    putStrLn $ "Input B:  " ++ show (V.toList inputB)
    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show ([11.0, 22.0, 33.0, 44.0] :: [Float])

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
