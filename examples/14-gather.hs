{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 14: Gather rows from a matrix.
--
-- Input:  3x4 matrix
-- Indices: [0, 2] (gather rows 0 and 2)
-- Output: 2x4 matrix
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-gather

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
    putStrLn "=== Example 14: Gather ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[2, 4] @'F32 "main"
            [ FuncArg "operand" (TensorType [3, 4] F32)
            , FuncArg "indices" (TensorType [2, 1] I64)
            ]
            $ do
                operand <- arg @'[3, 4] @'F32
                indices <- arg @'[2, 1] @'I64
                gather operand indices [1] [0] [0] 1 [1, 4]

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    -- Input: 3x4 matrix (values 1..12)
    let inputOperand = V.fromList
            [ 1,  2,  3,  4
            , 5,  6,  7,  8
            , 9, 10, 11, 12
            ] :: V.Vector Float

    -- Indices: gather rows 0 and 2
    let inputIndices = V.fromList [0, 2] :: V.Vector Int64

    bufOperand <- toDeviceF32 api client inputOperand [3, 4]
    bufIndices <- toDevice api client inputIndices [2, 1] bufferTypeS64

    [bufY] <- execute api exec [bufOperand, bufIndices]
    result <- fromDeviceF32 api bufY 8

    let expected = [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
