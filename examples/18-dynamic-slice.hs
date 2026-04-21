{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 18: Dynamic slice using runtime-computed start indices.
--
-- Input: 4x4 matrix
-- Start indices: [1, 2]
-- Slice sizes: [2, 2]
-- Output: 2x2 matrix
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-dynamic-slice

module Main where

import Data.Int (Int64)
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
    putStrLn "=== Example 18: Dynamic Slice ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
            [ FuncArg "operand" (TensorType [4, 4] F32)
            , FuncArg "start0"  (TensorType [] I64)
            , FuncArg "start1"  (TensorType [] I64)
            ]
            $ do
                operand <- arg @'[4, 4] @'F32
                start0  <- arg @'[] @'I64
                start1  <- arg @'[] @'I64
                dynamicSlice operand [start0, start1] [2, 2]

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    let inputOperand = V.fromList
            [ 1,  2,  3,  4
            , 5,  6,  7,  8
            , 9, 10, 11, 12
            , 13, 14, 15, 16
            ] :: V.Vector Float

    let inputStart0 = V.fromList [1] :: V.Vector Int64
    let inputStart1 = V.fromList [2] :: V.Vector Int64

    bufOperand <- toDeviceF32 api client inputOperand [4, 4]
    bufStart0  <- toDevice api client inputStart0 [] bufferTypeS64
    bufStart1  <- toDevice api client inputStart1 [] bufferTypeS64

    [bufY] <- execute api exec [bufOperand, bufStart0, bufStart1]
    result <- fromDeviceF32 api bufY 4

    -- Starting at [1,2], slice 2x2:
    -- [ 7,  8]
    -- [11, 12]
    let expected = [7.0, 8.0, 11.0, 12.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
