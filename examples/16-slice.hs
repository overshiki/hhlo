{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 16: Slice a sub-array from a tensor.
--
-- Input: 4x4 matrix
-- Slice: start=[0,1], limit=[2,4], stride=[1,1]
-- Output: 2x3 matrix (rows 0..1, cols 1..3)
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-slice

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
    putStrLn "=== Example 16: Slice ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[2, 3] @'F32 "main"
            [ FuncArg "operand" (TensorType [4, 4] F32)
            ]
            $ do
                operand <- arg @'[4, 4] @'F32
                slice operand [0, 1] [2, 4] [1, 1]

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    let inputOperand = V.fromList
            [ 1,  2,  3,  4
            , 5,  6,  7,  8
            , 9, 10, 11, 12
            , 13, 14, 15, 16
            ] :: V.Vector Float

    bufOperand <- toDeviceF32 api client inputOperand [4, 4]

    [bufY] <- execute api exec [bufOperand]
    result <- fromDeviceF32 api bufY 6

    let expected = [2.0, 3.0, 4.0, 6.0, 7.0, 8.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
