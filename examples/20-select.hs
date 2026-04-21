{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 20: Element-wise selection (select).
--
-- Input predicate: [True, False, True, False]
-- Input on-true:   [1.0, 2.0, 3.0, 4.0]
-- Input on-false:  [10.0, 20.0, 30.0, 40.0]
-- Output:          [1.0, 20.0, 3.0, 40.0]
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-select

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
    putStrLn "=== Example 20: Select ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[4] @'F32 "main"
            [ FuncArg "pred"    (TensorType [4] Bool)
            , FuncArg "onTrue"  (TensorType [4] F32)
            , FuncArg "onFalse" (TensorType [4] F32)
            ]
            $ do
                pred    <- arg @'[4] @'Bool
                onTrue  <- arg @'[4] @'F32
                onFalse <- arg @'[4] @'F32
                select pred onTrue onFalse

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    let inputPred    = V.fromList [1, 0, 1, 0] :: V.Vector Word8
    let inputOnTrue  = V.fromList [1.0, 2.0, 3.0, 4.0] :: V.Vector Float
    let inputOnFalse = V.fromList [10.0, 20.0, 30.0, 40.0] :: V.Vector Float

    bufPred    <- toDevice api client inputPred [4] bufferTypePred
    bufOnTrue  <- toDeviceF32 api client inputOnTrue [4]
    bufOnFalse <- toDeviceF32 api client inputOnFalse [4]

    [bufY] <- execute api exec [bufPred, bufOnTrue, bufOnFalse]
    result <- fromDeviceF32 api bufY 4

    let expected = [1.0, 20.0, 3.0, 40.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
