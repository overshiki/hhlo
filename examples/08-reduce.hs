{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 8: Reduction — sum all elements of a 2x3 matrix.
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-reduce

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
    putStrLn "=== Example 8: Reduction (sum all elements) ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- Build program: sum all elements of a 2x3 matrix
    let modu = moduleFromBuilder @'[] @'F32 "main"
            [ FuncArg "x" (TensorType [2, 3] F32)
            ]
            $ do
                x <- arg @'[2, 3] @'F32
                reduceSum x

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    -- Compile and run
    exec <- compile api client (render modu)

    let inputX = V.fromList [1, 2, 3, 4, 5, 6] :: V.Vector Float
    bufX <- toDeviceF32 api client inputX [2, 3]

    [bufY] <- execute api exec [bufX]
    result <- fromDeviceF32 api bufY 1

    -- Expected: 1+2+3+4+5+6 = 21
    let expected = [21.0] :: [Float]

    putStrLn $ "Input:    [1,2,3,4,5,6]"
    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected

    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
