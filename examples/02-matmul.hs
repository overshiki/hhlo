{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 2: Matrix multiplication.
--
-- Computes C = A × B where A is 2x3 and B is 3x2.
-- Result C is 2x2.

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
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

main :: IO ()
main = do
    putStrLn "=== Example 2: Matrix Multiplication ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- Build matmul program: C = A @ B
    let modu = moduleFromBuilder @'[2,2] @'F32 "main"
            [ FuncArg "a" (TensorType [2, 3] F32)
            , FuncArg "b" (TensorType [3, 2] F32)
            ]
            $ do
                a <- arg @'[2,3] @'F32
                b <- arg @'[3,2] @'F32
                c <- matmul a b
                return c

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    -- A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    -- B = [[7, 8], [9, 10], [11, 12]]  (3x2)
    -- C = [[58, 64], [139, 154]]  (2x2)
    let inputA = V.fromList [1, 2, 3, 4, 5, 6] :: V.Vector Float
        inputB = V.fromList [7, 8, 9, 10, 11, 12] :: V.Vector Float
    bufA <- toDeviceF32 api client inputA [2, 3]
    bufB <- toDeviceF32 api client inputB [3, 2]

    [bufC] <- execute api exec [bufA, bufB]
    result <- fromDeviceF32 api bufC 4

    putStrLn $ "A (2x3): " ++ show (V.toList inputA)
    putStrLn $ "B (3x2): " ++ show (V.toList inputB)
    putStrLn $ "C (2x2): " ++ show (V.toList result)
    putStrLn $ "Expected: [58.0,64.0,139.0,154.0]"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
