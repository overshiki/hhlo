{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 11: Batch Normalization (inference).
--
-- With mean=0, variance=1, scale=1, offset=0, the output equals the input
-- (modulo epsilon).
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-batch-norm

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
    putStrLn "=== Example 11: Batch Normalization (Inference) ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[2, 2, 2, 2] @'F32 "main"
            [ FuncArg "x" (TensorType [2, 2, 2, 2] F32)
            , FuncArg "scale"   (TensorType [2] F32)
            , FuncArg "offset"  (TensorType [2] F32)
            , FuncArg "mean"    (TensorType [2] F32)
            , FuncArg "variance" (TensorType [2] F32)
            ]
            $ do
                x  <- arg @'[2, 2, 2, 2] @'F32
                sc <- arg @'[2] @'F32
                off <- arg @'[2] @'F32
                mn <- arg @'[2] @'F32
                var <- arg @'[2] @'F32
                batchNormInference x sc off mn var

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    -- Input: 2x2x2x2 tensor
    let inputX = V.fromList [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] :: V.Vector Float
        scale  = V.fromList [1.0, 1.0] :: V.Vector Float
        offset = V.fromList [0.0, 0.0] :: V.Vector Float
        mean   = V.fromList [0.0, 0.0] :: V.Vector Float
        variance = V.fromList [1.0, 1.0] :: V.Vector Float

    bufX  <- toDeviceF32 api client inputX [2, 2, 2, 2]
    bufSc <- toDeviceF32 api client scale [2]
    bufOff <- toDeviceF32 api client offset [2]
    bufMn <- toDeviceF32 api client mean [2]
    bufVar <- toDeviceF32 api client variance [2]

    [bufY] <- execute api exec [bufX, bufSc, bufOff, bufMn, bufVar]
    result <- fromDeviceF32 api bufY 16

    -- Expected: approximately equal to input (since mean=0, var=1, scale=1, offset=0)
    let expected = V.toList inputX

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if approxEq (V.toList result) expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
    approxEq xs ys = all (\(x, y) -> abs (x - y) < 1e-4) (zip xs ys)
