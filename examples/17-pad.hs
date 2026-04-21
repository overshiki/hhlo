{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 17: Pad a tensor with a padding value.
--
-- Input: 2x2 matrix
-- Pad:  low=[1,0], high=[0,2], interior=[1,1]
-- Output: 4x5 matrix (with padding between elements and edges)
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-pad

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
    putStrLn "=== Example 17: Pad ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[4, 5] @'F32 "main"
            [ FuncArg "operand" (TensorType [2, 2] F32)
            ]
            $ do
                operand <- arg @'[2, 2] @'F32
                padValue <- constant @'[] @'F32 0.0
                pad operand padValue [1, 0] [0, 2] [1, 1]

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    let inputOperand = V.fromList
            [ 1, 2
            , 3, 4
            ] :: V.Vector Float

    bufOperand <- toDeviceF32 api client inputOperand [2, 2]

    [bufY] <- execute api exec [bufOperand]
    result <- fromDeviceF32 api bufY 20

    -- Expected layout (4 rows x 5 cols):
    -- [0, 0, 0, 0, 0]
    -- [1, 0, 2, 0, 0]
    -- [0, 0, 0, 0, 0]
    -- [3, 0, 4, 0, 0]
    let expected = [ 0.0, 0.0, 0.0, 0.0, 0.0
                   , 1.0, 0.0, 2.0, 0.0, 0.0
                   , 0.0, 0.0, 0.0, 0.0, 0.0
                   , 3.0, 0.0, 4.0, 0.0, 0.0
                   ] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
