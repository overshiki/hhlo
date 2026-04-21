{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 13: Conditional (if-then-else) selecting between two vectors.
--
-- The predicate is passed as a host argument (tensor<i1>).
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-conditional

module Main where

import Data.Word (Word8)
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
    putStrLn "=== Example 13: Conditional ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[3] @'F32 "main"
            [ FuncArg "pred" (TensorType [] Bool)
            ]
            $ do
                predVal <- arg @'[] @'Bool
                trueVal  <- constant @'[3] @'F32 1.0
                falseVal <- constant @'[3] @'F32 2.0
                conditional predVal (return trueVal) (return falseVal)

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    -- Predicate = true (1)
    let predData = V.fromList [1] :: V.Vector Word8
    bufPred <- toDevice api client predData [1] bufferTypePred

    [bufY] <- execute api exec [bufPred]
    result <- fromDeviceF32 api bufY 3

    let expected = [1.0, 1.0, 1.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
