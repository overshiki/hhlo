{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 32: Random bit generator (Threefry algorithm).
--
-- Takes a 2-element UI64 state tensor, generates a new state and
-- a 2x2 tensor of random UI64 bits.
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-rng-bit-generator

module Main where

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
    putStrLn "=== Example 32: RNG Bit Generator (Threefry) ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder2 @'[2] @'UI64 @'[2,2] @'UI64 "main"
            [ FuncArg "state" (TensorType [2] UI64) ]
            $ do
                state <- arg @'[2] @'UI64
                (newState, out) <- rngBitGenerator state
                returnTuple2 newState out

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    putStrLn "\nAttempting to compile and execute..."
    exec <- compile api client (render modu)

    let inputState = V.fromList [123456789 :: Int64, 987654321 :: Int64]
    bufState <- toDevice api client inputState [2] bufferTypeS64
    [bufNewState, bufOut] <- execute api exec [bufState]

    resultState <- fromDevice api bufNewState 2 :: IO (V.Vector Int64)
    resultOut   <- fromDevice api bufOut 4 :: IO (V.Vector Int64)

    putStrLn $ "Input state:  " ++ show (V.toList inputState)
    putStrLn $ "New state:    " ++ show (V.toList resultState)
    putStrLn $ "Output (2x2): " ++ show (V.toList resultOut)

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
