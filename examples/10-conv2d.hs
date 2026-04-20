{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 10: 2-D Convolution (NHWC format).
--
-- Input:  1x4x4x1  (batch=1, height=4, width=4, channels=1)
-- Kernel: 3x3x1x1  (kernel_h=3, kernel_w=3, in_ch=1, out_ch=1)
-- Output: 1x2x2x1  (no padding, stride=1)
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-conv2d

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
    putStrLn "=== Example 10: Conv2D ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
            [ FuncArg "x" (TensorType [1, 4, 4, 1] F32)
            , FuncArg "k" (TensorType [3, 3, 1, 1] F32)
            ]
            $ do
                x <- arg @'[1, 4, 4, 1] @'F32
                k <- arg @'[3, 3, 1, 1] @'F32
                conv2d x k

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    -- Input: 1x4x4x1 image (values 1..16)
    let inputX = V.fromList
            [ 1,  2,  3,  4
            , 5,  6,  7,  8
            , 9,  10, 11, 12
            , 13, 14, 15, 16
            ] :: V.Vector Float

    -- Kernel: 3x3x1x1 all ones
    let inputK = V.fromList (replicate 9 1.0) :: V.Vector Float

    bufX <- toDeviceF32 api client inputX [1, 4, 4, 1]
    bufK <- toDeviceF32 api client inputK [3, 3, 1, 1]

    [bufY] <- execute api exec [bufX, bufK]
    result <- fromDeviceF32 api bufY 4

    -- Expected (all-ones kernel is a sum filter):
    -- top-left     = 1+2+3+5+6+7+9+10+11  = 54
    -- top-right    = 2+3+4+6+7+8+10+11+12 = 63
    -- bottom-left  = 5+6+7+9+10+11+13+14+15 = 90
    -- bottom-right = 6+7+8+10+11+12+14+15+16 = 99
    let expected = [54.0, 63.0, 90.0, 99.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
