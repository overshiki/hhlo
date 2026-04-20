{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 9: Softmax on 1-D and 2-D tensors.
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-softmax

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
    putStrLn "=== Example 9: Softmax ==="

    -- Load plugin and create client
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- --- 1-D Softmax ---
    putStrLn "\n--- 1-D Softmax ---"
    let mod1D = moduleFromBuilder @'[4] @'F32 "main"
            [ FuncArg "x" (TensorType [4] F32) ]
            $ do
                x <- arg @'[4] @'F32
                softmax1D x

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render mod1D)

    exec1D <- compile api client (render mod1D)
    let input1D = V.fromList [1.0, 2.0, 3.0, 4.0] :: V.Vector Float
    buf1D <- toDeviceF32 api client input1D [4]
    [out1D] <- execute api exec1D [buf1D]
    result1D <- fromDeviceF32 api out1D 4

    -- Expected: exp([1,2,3,4]) / sum(exp([1,2,3,4]))
    let exps = map exp [1.0, 2.0, 3.0, 4.0] :: [Float]
        sumExps = sum exps
        expected1D = map (/ sumExps) exps
    putStrLn $ "Result:   " ++ show (V.toList result1D)
    putStrLn $ "Expected: " ++ show expected1D

    -- --- 2-D Softmax ---
    putStrLn "\n--- 2-D Softmax (batched) ---"
    let mod2D = moduleFromBuilder @'[2, 3] @'F32 "main"
            [ FuncArg "x" (TensorType [2, 3] F32) ]
            $ do
                x <- arg @'[2, 3] @'F32
                softmax2D x

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render mod2D)

    exec2D <- compile api client (render mod2D)
    let input2D = V.fromList [1.0, 2.0, 3.0, 1.0, 2.0, 3.0] :: V.Vector Float
    buf2D <- toDeviceF32 api client input2D [2, 3]
    [out2D] <- execute api exec2D [buf2D]
    result2D <- fromDeviceF32 api out2D 6

    -- Each row independently: exp([1,2,3]) / sum(exp([1,2,3]))
    let exps2 = map exp [1.0, 2.0, 3.0] :: [Float]
        sumExps2 = sum exps2
        expectedRow = map (/ sumExps2) exps2
        expected2D = expectedRow ++ expectedRow
    putStrLn $ "Result:   " ++ show (V.toList result2D)
    putStrLn $ "Expected: " ++ show expected2D

    if approxEq (V.toList result1D) expected1D && approxEq (V.toList result2D) expected2D
        then putStrLn "\n✓ PASS"
        else putStrLn "\n✗ FAIL"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
    approxEq xs ys = all (\(x, y) -> abs (x - y) < 1e-4) (zip xs ys)
