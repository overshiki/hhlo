{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 6: Batched MLP (Multi-Layer Perceptron) forward pass.
--
-- Architecture: input[batch,4] -> linear(4,8) -> relu -> linear(8,2) -> output[batch,2]
--
-- This example demonstrates 'linearBatched' which uses 'broadcast_in_dim'
-- to broadcast the 1-D bias to the batched output shape.
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-mlp-batched

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
import HHLO.Runtime.Execute

main :: IO ()
main = do
    putStrLn "=== Example 6: Batched MLP Forward Pass ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- Build batched MLP program:
    --   h  = linearBatched(x, W1, b1)   -- [batch,4] -> [batch,8]
    --   a  = relu(h)                    -- [batch,8]
    --   y  = linearBatched(a, W2, b2)   -- [batch,8] -> [batch,2]
    let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
            [ FuncArg "x"  (TensorType [2, 4]   F32)
            , FuncArg "w1" (TensorType [4, 8]   F32)
            , FuncArg "b1" (TensorType [8]      F32)
            , FuncArg "w2" (TensorType [8, 2]   F32)
            , FuncArg "b2" (TensorType [2]      F32)
            ]
            $ do
                x  <- arg @'[2, 4]   @'F32
                w1 <- arg @'[4, 8]   @'F32
                b1 <- arg @'[8]      @'F32
                w2 <- arg @'[8, 2]   @'F32
                b2 <- arg @'[2]      @'F32
                h  <- linearBatched x w1 b1
                a  <- relu h
                linearBatched a w2 b2

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    -- Compile and run
    exec <- compile api client (render modu)

    -- Host data: batch of 2 samples, each with 4 features
    let inputX = V.fromList
            [ 1, 2, 3, 4    -- sample 0
            , 5, 6, 7, 8    -- sample 1
            ] :: V.Vector Float

    -- W1: 4x8 — identity-like so hidden = [x, 0, 0, 0, 0]
    let w1Vals = V.fromList
            [ 1,0,0,0,0,0,0,0
            , 0,1,0,0,0,0,0,0
            , 0,0,1,0,0,0,0,0
            , 0,0,0,1,0,0,0,0
            ] :: V.Vector Float

    -- b1: zeros
    let b1Vals = V.fromList [0,0,0,0,0,0,0,0] :: V.Vector Float

    -- W2: 8x2 — identity-like so output = [h0, h1]
    let w2Vals = V.fromList
            [ 1,0
            , 0,1
            , 0,0
            , 0,0
            , 0,0
            , 0,0
            , 0,0
            , 0,0
            ] :: V.Vector Float

    -- b2: zeros
    let b2Vals = V.fromList [0,0] :: V.Vector Float

    bufX  <- toDeviceF32 api client inputX [2, 4]
    bufW1 <- toDeviceF32 api client w1Vals [4, 8]
    bufB1 <- toDeviceF32 api client b1Vals [8]
    bufW2 <- toDeviceF32 api client w2Vals [8, 2]
    bufB2 <- toDeviceF32 api client b2Vals [2]

    [bufY] <- execute api exec [bufX, bufW1, bufB1, bufW2, bufB2]
    result <- fromDeviceF32 api bufY 4

    -- Expected:
    -- sample 0: h = [1,2,3,4,0,0,0,0] -> a = [1,2,3,4,0,0,0,0] -> y = [1,2]
    -- sample 1: h = [5,6,7,8,0,0,0,0] -> a = [5,6,7,8,0,0,0,0] -> y = [5,6]
    let expected = [1.0, 2.0, 5.0, 6.0] :: [Float]

    putStrLn $ "Input shape:    [2, 4]"
    putStrLn $ "Output shape:   [2, 2]"
    putStrLn $ "Result:         " ++ show (V.toList result)
    putStrLn $ "Expected:       " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
