{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 26: UNet for image segmentation.
--
-- Toy-sized UNet for fast CPU execution:
--   Input: 16x16x1  →  Output: 16x16x2 (2-class segmentation)
--   2 encoder/decoder stages with skip connections
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-unet

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
    putStrLn "=== Example 26: UNet (toy, 16x16 input) ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[1, 16, 16, 2] @'F32 "main"
            [ FuncArg "input" (TensorType [1, 16, 16, 1] F32)
            ]
            $ do
                x <- arg @'[1, 16, 16, 1] @'F32

                -- ========== Encoder ==========

                -- Encoder stage 1: 1 -> 16 channels
                wE1a <- constant @'[3, 3, 1, 16] @'F32 0.01
                e1 <- conv2dWithPadding @1 @16 @16 @1 @16 @3 @3 @16 @16 [1, 1] [[1, 1], [1, 1]] x wE1a
                e1 <- relu e1
                wE1b <- constant @'[3, 3, 16, 16] @'F32 0.01
                e1 <- conv2dWithPadding @1 @16 @16 @16 @16 @3 @3 @16 @16 [1, 1] [[1, 1], [1, 1]] e1 wE1b
                e1 <- relu e1
                -- Skip1 = e1 (shape [1,16,16,16])

                -- Downsample: maxPool 2x2/2
                d1 <- maxPool [2, 2] [2, 2] [[0, 0], [0, 0]] e1

                -- Encoder stage 2: 16 -> 32 channels
                wE2a <- constant @'[3, 3, 16, 32] @'F32 0.01
                e2 <- conv2dWithPadding @1 @8 @8 @16 @32 @3 @3 @8 @8 [1, 1] [[1, 1], [1, 1]] d1 wE2a
                e2 <- relu e2
                wE2b <- constant @'[3, 3, 32, 32] @'F32 0.01
                e2 <- conv2dWithPadding @1 @8 @8 @32 @32 @3 @3 @8 @8 [1, 1] [[1, 1], [1, 1]] e2 wE2b
                e2 <- relu e2
                -- Skip2 = e2 (shape [1,8,8,32])

                -- Downsample: maxPool 2x2/2
                d2 <- maxPool [2, 2] [2, 2] [[0, 0], [0, 0]] e2

                -- ========== Bottleneck ==========
                wBa <- constant @'[3, 3, 32, 64] @'F32 0.01
                b <- conv2dWithPadding @1 @4 @4 @32 @64 @3 @3 @4 @4 [1, 1] [[1, 1], [1, 1]] d2 wBa
                b <- relu b
                wBb <- constant @'[3, 3, 64, 64] @'F32 0.01
                b <- conv2dWithPadding @1 @4 @4 @64 @64 @3 @3 @4 @4 [1, 1] [[1, 1], [1, 1]] b wBb
                b <- relu b

                -- ========== Decoder ==========

                -- Decoder stage 2: upsample 4x4 -> 8x8, concat with skip2
                -- Transpose conv: [1,4,4,64] -> [1,8,8,32]
                wD2up <- constant @'[2, 2, 32, 64] @'F32 0.01
                u2 <- transposeConvolution [1, 2, 2, 1] [[1, 1], [1, 1]] b wD2up
                -- Concatenate with skip2 along channel dim (axis 3)
                u2 <- concatenate2 @'[1, 8, 8, 32] @'[1, 8, 8, 32] @'[1, 8, 8, 64] @'F32 3 u2 e2
                wD2a <- constant @'[3, 3, 64, 32] @'F32 0.01
                u2 <- conv2dWithPadding @1 @8 @8 @64 @32 @3 @3 @8 @8 [1, 1] [[1, 1], [1, 1]] u2 wD2a
                u2 <- relu u2
                wD2b <- constant @'[3, 3, 32, 32] @'F32 0.01
                u2 <- conv2dWithPadding @1 @8 @8 @32 @32 @3 @3 @8 @8 [1, 1] [[1, 1], [1, 1]] u2 wD2b
                u2 <- relu u2

                -- Decoder stage 1: upsample 8x8 -> 16x16, concat with skip1
                wD1up <- constant @'[2, 2, 16, 32] @'F32 0.01
                u1 <- transposeConvolution [1, 2, 2, 1] [[1, 1], [1, 1]] u2 wD1up
                -- Concatenate with skip1 along channel dim (axis 3)
                u1 <- concatenate2 @'[1, 16, 16, 16] @'[1, 16, 16, 16] @'[1, 16, 16, 32] @'F32 3 u1 e1
                wD1a <- constant @'[3, 3, 32, 16] @'F32 0.01
                u1 <- conv2dWithPadding @1 @16 @16 @32 @16 @3 @3 @16 @16 [1, 1] [[1, 1], [1, 1]] u1 wD1a
                u1 <- relu u1
                wD1b <- constant @'[3, 3, 16, 16] @'F32 0.01
                u1 <- conv2dWithPadding @1 @16 @16 @16 @16 @3 @3 @16 @16 [1, 1] [[1, 1], [1, 1]] u1 wD1b
                u1 <- relu u1

                -- Final 1x1 conv: 16 -> 2 channels
                wOut <- constant @'[1, 1, 16, 2] @'F32 0.01
                conv2dWithPadding @1 @16 @16 @16 @2 @1 @1 @16 @16 [1, 1] [[0, 0], [0, 0]] u1 wOut

    putStrLn "Generated MLIR (first 20 lines):"
    let lines_ = T.lines (render modu)
    mapM_ (putStrLn . T.unpack) (take 20 lines_)
    putStrLn "  ..."

    exec <- compile api client (render modu)

    let input = V.fromList [0.5 | _ <- [1..256]] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 16, 16, 1]

    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 512

    putStrLn $ "Output shape: [1, 16, 16, 2]"
    putStrLn $ "Output (first 8): " ++ show (take 8 (V.toList result))
    putStrLn $ "Output (last 8):  " ++ show (drop 504 (V.toList result))
    putStrLn "✓ PASS (executed successfully)"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
