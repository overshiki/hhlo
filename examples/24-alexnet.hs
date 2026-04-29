{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 24: AlexNet-style CNN.
--
-- Toy-sized AlexNet for fast CPU execution:
--   Input: 16x16x3  →  Output: 10 classes
--   3 conv groups + 2 FC layers
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-alexnet

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
    putStrLn "=== Example 24: AlexNet (toy, 16x16 input) ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[1, 10] @'F32 "main"
            [ FuncArg "input" (TensorType [1, 16, 16, 3] F32)
            ]
            $ do
                x <- arg @'[1, 16, 16, 3] @'F32

                -- Conv 3x3/1, 32 channels
                w1 <- constant @'[3, 3, 3, 32] @'F32 0.01
                x <- conv2dWithPadding @1 @16 @16 @3 @32 @3 @3 @16 @16 (v2 1 1) (p2 (1,1) (1,1)) x w1
                x <- relu x
                -- MaxPool 2x2/2
                x <- maxPool (v2 2 2) (v2 2 2) (p2 (0,0) (0,0)) x

                -- Conv 3x3/1, 64 channels
                w2 <- constant @'[3, 3, 32, 64] @'F32 0.01
                x <- conv2dWithPadding @1 @8 @8 @32 @64 @3 @3 @8 @8 (v2 1 1) (p2 (1,1) (1,1)) x w2
                x <- relu x
                -- MaxPool 2x2/2
                x <- maxPool (v2 2 2) (v2 2 2) (p2 (0,0) (0,0)) x

                -- Conv 3x3/1, 128 channels
                w3 <- constant @'[3, 3, 64, 128] @'F32 0.01
                x <- conv2dWithPadding @1 @4 @4 @64 @128 @3 @3 @4 @4 (v2 1 1) (p2 (1,1) (1,1)) x w3
                x <- relu x

                -- Conv 3x3/1, 128 channels
                w4 <- constant @'[3, 3, 128, 128] @'F32 0.01
                x <- conv2dWithPadding @1 @4 @4 @128 @128 @3 @3 @4 @4 (v2 1 1) (p2 (1,1) (1,1)) x w4
                x <- relu x
                -- MaxPool 2x2/2
                x <- maxPool (v2 2 2) (v2 2 2) (p2 (0,0) (0,0)) x
                let x' :: Tensor '[1, 2, 2, 128] 'F32
                    x' = x

                -- Flatten: [1, 2, 2, 128] -> [1, 512]
                x <- reshape @'[1, 2, 2, 128] @'[1, 512] x'

                -- FC 512 -> 128
                wFC1 <- constant @'[512, 128] @'F32 0.01
                bFC1 <- constant @'[128] @'F32 0.0
                x <- linearBatched x wFC1 bFC1
                x <- relu x

                -- FC 128 -> 10
                wFC2 <- constant @'[128, 10] @'F32 0.01
                bFC2 <- constant @'[10] @'F32 0.0
                x <- linearBatched x wFC2 bFC2

                softmax2D x

    putStrLn "Generated MLIR (first 20 lines):"
    let lines_ = T.lines (render modu)
    mapM_ (putStrLn . T.unpack) (take 20 lines_)
    putStrLn "  ..."

    exec <- compile api client (render modu)

    let input = V.fromList [0.5 | _ <- [1..768]] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 16, 16, 3]

    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 10

    putStrLn $ "Output (10 class logits after softmax): " ++ show (V.toList result)
    putStrLn $ "Sum of probabilities: " ++ show (sum (V.toList result))
    putStrLn "✓ PASS (executed successfully)"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
