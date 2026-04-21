{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 15: Scatter updates into a zero vector (replace semantics).
--
-- Input:  [0, 0, 0, 0, 0]
-- Indices: [[1], [3]]
-- Updates: [10, 20]
-- Output: [0, 10, 0, 20, 0]
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-scatter

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
    putStrLn "=== Example 15: Scatter ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[5] @'F32 "main"
            [ FuncArg "input"   (TensorType [5] F32)
            , FuncArg "indices" (TensorType [2, 1] I64)
            , FuncArg "updates" (TensorType [2] F32)
            ]
            $ do
                input   <- arg @'[5]     @'F32
                indices <- arg @'[2, 1] @'I64
                updates <- arg @'[2]     @'F32
                scatter input indices updates
                    (\_ upd -> return upd)   -- replace semantics
                    [] [0] [0] 1

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    -- Input: 5 zeros
    let inputData = V.fromList [0.0, 0.0, 0.0, 0.0, 0.0] :: V.Vector Float

    -- Indices: positions 1 and 3
    let indexData = V.fromList [1, 3] :: V.Vector Int64

    -- Updates: 10 and 20
    let updateData = V.fromList [10.0, 20.0] :: V.Vector Float

    bufInput   <- toDeviceF32 api client inputData [5]
    bufIndices <- toDevice api client indexData [2, 1] bufferTypeS64
    bufUpdates <- toDeviceF32 api client updateData [2]

    [bufY] <- execute api exec [bufInput, bufIndices, bufUpdates]
    result <- fromDeviceF32 api bufY 5

    let expected = [0.0, 10.0, 0.0, 20.0, 0.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
