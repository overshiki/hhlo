{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 21: Element-wise map over all dimensions.
--
-- Input: [1.0, 2.0, 3.0, 4.0]
-- Computation: x -> x * x + 1
-- Output: [2.0, 5.0, 10.0, 17.0]
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-map

module Main where

import Prelude hiding (map)

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
    putStrLn "=== Example 21: Map ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[4] @'F32 "main"
            [ FuncArg "x" (TensorType [4] F32)
            ]
            $ do
                x <- arg @'[4] @'F32
                -- Map over all dimensions [0] for a 1-D tensor.
                -- Computation: a -> a * a + 1
                map [x] [0] $ \[a] -> do
                    sq <- multiply a a
                    one <- constant @'[] @'F32 1.0
                    add sq one

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    let inputX = V.fromList [1.0, 2.0, 3.0, 4.0] :: V.Vector Float

    bufX <- toDeviceF32 api client inputX [4]

    [bufY] <- execute api exec [bufX]
    result <- fromDeviceF32 api bufY 4

    let expected = [2.0, 5.0, 10.0, 17.0] :: [Float]

    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
