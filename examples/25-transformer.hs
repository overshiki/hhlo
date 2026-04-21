{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 25: Transformer encoder (single layer).
--
-- Toy-sized Transformer for fast CPU execution:
--   batch=1, seq=4, d_model=16, num_heads=2, head_dim=8, ff_dim=32
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-transformer

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

-- Multi-head self-attention
-- Input: [batch, seq, d_model]
-- WQ/WK/WV: [d_model, d_model]
-- WO: [d_model, d_model]
multiHeadAttention :: Tensor '[1, 4, 16] 'F32
                   -> Tensor '[16, 16] 'F32
                   -> Tensor '[16, 16] 'F32
                   -> Tensor '[16, 16] 'F32
                   -> Tensor '[16, 16] 'F32
                   -> Builder (Tensor '[1, 4, 16] 'F32)
multiHeadAttention x wq wk wv wo = do
    -- Linear projections: [1,4,16] @ [16,16] -> [1,4,16]
    q <- dotGeneral @'[1, 4, 16] @'[16, 16] @'[1, 4, 16] @'F32 [] [] [2] [0] x wq
    k <- dotGeneral @'[1, 4, 16] @'[16, 16] @'[1, 4, 16] @'F32 [] [] [2] [0] x wk
    v <- dotGeneral @'[1, 4, 16] @'[16, 16] @'[1, 4, 16] @'F32 [] [] [2] [0] x wv

    -- Split into heads: [1,4,16] -> [1,4,2,8] -> [1,2,4,8]
    q <- reshape @'[1, 4, 16] @'[1, 4, 2, 8] q
    q <- transpose @'[1, 4, 2, 8] @'[1, 2, 4, 8] [0, 2, 1, 3] q
    k <- reshape @'[1, 4, 16] @'[1, 4, 2, 8] k
    k <- transpose @'[1, 4, 2, 8] @'[1, 2, 4, 8] [0, 2, 1, 3] k
    v <- reshape @'[1, 4, 16] @'[1, 4, 2, 8] v
    v <- transpose @'[1, 4, 2, 8] @'[1, 2, 4, 8] [0, 2, 1, 3] v

    -- Attention scores: [1,2,4,8] @ [1,2,8,4] -> [1,2,4,4]
    kt <- transpose @'[1, 2, 4, 8] @'[1, 2, 8, 4] [0, 1, 3, 2] k
    scores <- dotGeneral @'[1, 2, 4, 8] @'[1, 2, 8, 4] @'[1, 2, 4, 4] @'F32 [0, 1] [0, 1] [3] [2] q kt
    scale <- constant @'[] @'F32 (1.0 / sqrt 8.0)
    scaleBC <- broadcastWithDims @'[] @'[1, 2, 4, 4] [] scale
    scores <- divide scores scaleBC
    weights <- softmax4D scores

    -- Apply to values: [1,2,4,4] @ [1,2,4,8] -> [1,2,4,8]
    out <- dotGeneral @'[1, 2, 4, 4] @'[1, 2, 4, 8] @'[1, 2, 4, 8] @'F32 [0, 1] [0, 1] [3] [2] weights v

    -- Merge heads: [1,2,4,8] -> [1,4,2,8] -> [1,4,16]
    out <- transpose @'[1, 2, 4, 8] @'[1, 4, 2, 8] [0, 2, 1, 3] out
    out <- reshape @'[1, 4, 2, 8] @'[1, 4, 16] out

    -- Output projection
    dotGeneral @'[1, 4, 16] @'[16, 16] @'[1, 4, 16] @'F32 [] [] [2] [0] out wo

-- 3-D linear: [batch, seq, inDim] @ [inDim, outDim] + [outDim]
linear3D :: Tensor '[1, 4, 16] 'F32 -> Tensor '[16, 32] 'F32 -> Tensor '[32] 'F32
         -> Builder (Tensor '[1, 4, 32] 'F32)
linear3D x w b = do
    y <- dotGeneral @'[1, 4, 16] @'[16, 32] @'[1, 4, 32] @'F32 [] [] [2] [0] x w
    b' <- broadcastWithDims @'[32] @'[1, 4, 32] [2] b
    add y b'

linear3D' :: Tensor '[1, 4, 32] 'F32 -> Tensor '[32, 16] 'F32 -> Tensor '[16] 'F32
          -> Builder (Tensor '[1, 4, 16] 'F32)
linear3D' x w b = do
    y <- dotGeneral @'[1, 4, 32] @'[32, 16] @'[1, 4, 16] @'F32 [] [] [2] [0] x w
    b' <- broadcastWithDims @'[16] @'[1, 4, 16] [2] b
    add y b'

-- Feed-forward network: linear -> gelu -> linear
feedForward :: Tensor '[1, 4, 16] 'F32
            -> Tensor '[16, 32] 'F32
            -> Tensor '[32] 'F32
            -> Tensor '[32, 16] 'F32
            -> Tensor '[16] 'F32
            -> Builder (Tensor '[1, 4, 16] 'F32)
feedForward x w1 b1 w2 b2 = do
    h <- linear3D x w1 b1
    h <- gelu h
    linear3D' h w2 b2

main :: IO ()
main = do
    putStrLn "=== Example 25: Transformer Encoder (toy, 1x4x16) ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[1, 4, 16] @'F32 "main"
            [ FuncArg "input" (TensorType [1, 4, 16] F32)
            ]
            $ do
                x <- arg @'[1, 4, 16] @'F32

                -- MHA weights
                wq <- constant @'[16, 16] @'F32 0.01
                wk <- constant @'[16, 16] @'F32 0.01
                wv <- constant @'[16, 16] @'F32 0.01
                wo <- constant @'[16, 16] @'F32 0.01

                -- MHA + residual
                attn <- multiHeadAttention x wq wk wv wo
                x <- add x attn

                -- LayerNorm
                gamma <- constant @'[16] @'F32 1.0
                beta  <- constant @'[16] @'F32 0.0
                x <- layerNorm x gamma beta

                -- FFN weights
                w1 <- constant @'[16, 32] @'F32 0.01
                b1 <- constant @'[32] @'F32 0.0
                w2 <- constant @'[32, 16] @'F32 0.01
                b2 <- constant @'[16] @'F32 0.0

                -- FFN + residual
                ffn <- feedForward x w1 b1 w2 b2
                x <- add x ffn

                -- Final layerNorm
                x <- layerNorm x gamma beta

                return x

    putStrLn "Generated MLIR (first 20 lines):"
    let lines_ = T.lines (render modu)
    mapM_ (putStrLn . T.unpack) (take 20 lines_)
    putStrLn "  ..."

    exec <- compile api client (render modu)

    let input = V.fromList [0.1 * fromIntegral i | i <- [1..64]] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 4, 16]

    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 64

    putStrLn $ "Output shape: [1, 4, 16]"
    putStrLn $ "Output (first 8): " ++ show (take 8 (V.toList result))
    putStrLn $ "Output (last 8):  " ++ show (drop 56 (V.toList result))
    putStrLn "✓ PASS (executed successfully)"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
