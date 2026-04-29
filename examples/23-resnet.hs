{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 23: ResNet-18-style CNN for CIFAR-10.
--
-- Toy-sized ResNet for fast CPU execution:
--   Input: 8x8x3  →  Output: 10 classes
--   3 stages with basic blocks + skip connections
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-resnet

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

-- ---------------------------------------------------------------------------
-- ResNet blocks (concrete shapes for 8x8 input)
-- ---------------------------------------------------------------------------

-- Initial conv: 3x3 stride 1 pad 1, 3 -> 16 channels
initialConv :: Tensor '[1, 8, 8, 3] 'F32 -> Builder (Tensor '[1, 8, 8, 16] 'F32)
initialConv x = do
    w <- constant @'[3, 3, 3, 16] @'F32 0.01
    conv2dWithPadding @1 @8 @8 @3 @16 @3 @3 @8 @8 (v2 1 1) (p2 (1,1) (1,1)) x w

-- Stage 1: 2 basic blocks, 16 channels, 8x8 (no downsampling)
stage1Block1 :: Tensor '[1, 8, 8, 16] 'F32 -> Builder (Tensor '[1, 8, 8, 16] 'F32)
stage1Block1 x = do
    w1 <- constant @'[3, 3, 16, 16] @'F32 0.01
    out <- conv2dWithPadding @1 @8 @8 @16 @16 @3 @3 @8 @8 (v2 1 1) (p2 (1,1) (1,1)) x w1
    out <- relu out
    w2 <- constant @'[3, 3, 16, 16] @'F32 0.01
    out <- conv2dWithPadding @1 @8 @8 @16 @16 @3 @3 @8 @8 (v2 1 1) (p2 (1,1) (1,1)) out w2
    add out x

stage1Block2 :: Tensor '[1, 8, 8, 16] 'F32 -> Builder (Tensor '[1, 8, 8, 16] 'F32)
stage1Block2 x = do
    w1 <- constant @'[3, 3, 16, 16] @'F32 0.01
    out <- conv2dWithPadding @1 @8 @8 @16 @16 @3 @3 @8 @8 (v2 1 1) (p2 (1,1) (1,1)) x w1
    out <- relu out
    w2 <- constant @'[3, 3, 16, 16] @'F32 0.01
    out <- conv2dWithPadding @1 @8 @8 @16 @16 @3 @3 @8 @8 (v2 1 1) (p2 (1,1) (1,1)) out w2
    add out x

-- Stage 2: 2 blocks, 16 -> 32 channels, 8x8 -> 4x4 (stride 2 on first)
stage2Block1 :: Tensor '[1, 8, 8, 16] 'F32 -> Builder (Tensor '[1, 4, 4, 32] 'F32)
stage2Block1 x = do
    -- Projection shortcut
    wSkip <- constant @'[1, 1, 16, 32] @'F32 0.01
    skip <- conv2dWithPadding @1 @8 @8 @16 @32 @1 @1 @4 @4 (v2 2 2) (p2 (0,0) (0,0)) x wSkip
    -- Main path
    w1 <- constant @'[3, 3, 16, 32] @'F32 0.01
    out <- conv2dWithPadding @1 @8 @8 @16 @32 @3 @3 @4 @4 (v2 2 2) (p2 (1,1) (1,1)) x w1
    out <- relu out
    w2 <- constant @'[3, 3, 32, 32] @'F32 0.01
    out <- conv2dWithPadding @1 @4 @4 @32 @32 @3 @3 @4 @4 (v2 1 1) (p2 (1,1) (1,1)) out w2
    add out skip

stage2Block2 :: Tensor '[1, 4, 4, 32] 'F32 -> Builder (Tensor '[1, 4, 4, 32] 'F32)
stage2Block2 x = do
    w1 <- constant @'[3, 3, 32, 32] @'F32 0.01
    out <- conv2dWithPadding @1 @4 @4 @32 @32 @3 @3 @4 @4 (v2 1 1) (p2 (1,1) (1,1)) x w1
    out <- relu out
    w2 <- constant @'[3, 3, 32, 32] @'F32 0.01
    out <- conv2dWithPadding @1 @4 @4 @32 @32 @3 @3 @4 @4 (v2 1 1) (p2 (1,1) (1,1)) out w2
    add out x

-- Stage 3: 2 blocks, 32 -> 64 channels, 4x4 -> 2x2 (stride 2 on first)
stage3Block1 :: Tensor '[1, 4, 4, 32] 'F32 -> Builder (Tensor '[1, 2, 2, 64] 'F32)
stage3Block1 x = do
    wSkip <- constant @'[1, 1, 32, 64] @'F32 0.01
    skip <- conv2dWithPadding @1 @4 @4 @32 @64 @1 @1 @2 @2 (v2 2 2) (p2 (0,0) (0,0)) x wSkip
    w1 <- constant @'[3, 3, 32, 64] @'F32 0.01
    out <- conv2dWithPadding @1 @4 @4 @32 @64 @3 @3 @2 @2 (v2 2 2) (p2 (1,1) (1,1)) x w1
    out <- relu out
    w2 <- constant @'[3, 3, 64, 64] @'F32 0.01
    out <- conv2dWithPadding @1 @2 @2 @64 @64 @3 @3 @2 @2 (v2 1 1) (p2 (1,1) (1,1)) out w2
    add out skip

stage3Block2 :: Tensor '[1, 2, 2, 64] 'F32 -> Builder (Tensor '[1, 2, 2, 64] 'F32)
stage3Block2 x = do
    w1 <- constant @'[3, 3, 64, 64] @'F32 0.01
    out <- conv2dWithPadding @1 @2 @2 @64 @64 @3 @3 @2 @2 (v2 1 1) (p2 (1,1) (1,1)) x w1
    out <- relu out
    w2 <- constant @'[3, 3, 64, 64] @'F32 0.01
    out <- conv2dWithPadding @1 @2 @2 @64 @64 @3 @3 @2 @2 (v2 1 1) (p2 (1,1) (1,1)) out w2
    add out x

-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------

main :: IO ()
main = do
    putStrLn "=== Example 23: ResNet-18 (toy, 8x8 input) ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[1, 10] @'F32 "main"
            [ FuncArg "input" (TensorType [1, 8, 8, 3] F32)
            ]
            $ do
                x <- arg @'[1, 8, 8, 3] @'F32

                -- Initial conv
                x <- initialConv x
                x <- relu x

                -- Stage 1
                x <- stage1Block1 x
                x <- stage1Block2 x

                -- Stage 2
                x <- stage2Block1 x
                x <- stage2Block2 x

                -- Stage 3
                x <- stage3Block1 x
                x <- stage3Block2 x

                -- Global avg pool
                x <- globalAvgPool x

                -- FC 64 -> 10
                wFC <- constant @'[64, 10] @'F32 0.01
                x <- matmul x wFC

                -- Softmax
                softmax2D x

    putStrLn "Generated MLIR (first 20 lines):"
    let lines_ = T.lines (render modu)
    mapM_ (putStrLn . T.unpack) (take 20 lines_)
    putStrLn "  ..."

    exec <- compile api client (render modu)

    let input = V.fromList [0.5 | _ <- [1..192]] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 8, 8, 3]

    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 10

    putStrLn $ "Output (10 class logits after softmax): " ++ show (V.toList result)
    putStrLn $ "Sum of probabilities: " ++ show (sum (V.toList result))
    putStrLn "✓ PASS (executed successfully)"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
