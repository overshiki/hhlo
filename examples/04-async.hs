{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 4: Demonstrate async execution with bufferReady / awaitBuffers.
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-async

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
import HHLO.Runtime.Buffer
import HHLO.Runtime.Async

main :: IO ()
main = do
    putStrLn "=== Example 4: Async Execution ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- Build program: relu(a) + b
    let modu = moduleFromBuilder @'[4] @'F32 "main"
            [ FuncArg "a" (TensorType [4] F32)
            , FuncArg "b" (TensorType [4] F32)
            ]
            $ do
                a <- arg
                b <- arg
                r <- relu a
                add r b

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    -- Compile and run
    exec <- compile api client (render modu)

    let inputA = V.fromList [-1, 2, -3, 4] :: V.Vector Float
        inputB = V.fromList [1, 1, 1, 1] :: V.Vector Float
    bufA <- toDeviceF32 api client inputA [4]
    bufB <- toDeviceF32 api client inputB [4]

    -- Execute asynchronously (returns immediately with valid handles)
    [bufC] <- executeAsync api exec [bufA, bufB]

    -- Poll readiness
    ready <- bufferReady api bufC
    putStrLn $ "Buffer ready immediately after execute? " ++ show ready

    -- Await explicitly
    awaitBuffers api [bufC]
    putStrLn "Buffer awaited successfully."

    result <- fromDeviceF32 api bufC 4
    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show ([1.0, 3.0, 1.0, 5.0] :: [Float])

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
