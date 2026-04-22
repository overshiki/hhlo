{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 33: Multi-value while loop — counter and accumulator.
--
-- This demonstrates whileLoop2, which carries two tensors through the loop.
-- The loop counts from 0 up to 3, accumulating a running sum.
--   counter: 0 -> 1 -> 2 -> 3
--   sum:     0 -> 1 -> 3 -> 6
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-multi-value-loop

module Main where

import Prelude hiding (compare)

import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Data.Int (Int64)
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
    putStrLn "=== Example 33: Multi-Value While Loop ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder2 @'[] @'I64 @'[] @'I64 "main"
            [ FuncArg "counter_init" (TensorType [] I64)
            , FuncArg "sum_init"     (TensorType [] I64)
            ]
            $ do
                counter0 <- arg @'[] @'I64
                sum0     <- arg @'[] @'I64
                result <- whileLoop2 counter0 sum0
                    (\c _s -> do
                        limit <- constant @'[] @'I64 3
                        compare c limit "LT")
                    (\c s -> do
                        one   <- constant @'[] @'I64 1
                        cNext <- add c one
                        sNext <- add s cNext
                        returnTuple2 cNext sNext)
                return result

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    putStrLn "\nExecuting..."
    exec <- compile api client (render modu)

    let inputCounter = V.fromList [0 :: Int64]
        inputSum     = V.fromList [0 :: Int64]
    bufCounter <- toDevice api client inputCounter [1] bufferTypeS64
    bufSum     <- toDevice api client inputSum     [1] bufferTypeS64

    [bufOutCounter, bufOutSum] <- execute api exec [bufCounter, bufSum]

    resultCounter <- fromDevice api bufOutCounter 1 :: IO (V.Vector Int64)
    resultSum     <- fromDevice api bufOutSum     1 :: IO (V.Vector Int64)

    putStrLn $ "Input counter:  " ++ show (V.toList inputCounter)
    putStrLn $ "Input sum:      " ++ show (V.toList inputSum)
    putStrLn $ "Output counter: " ++ show (V.toList resultCounter) ++ " (expected: [3])"
    putStrLn $ "Output sum:     " ++ show (V.toList resultSum)     ++ " (expected: [6])"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
