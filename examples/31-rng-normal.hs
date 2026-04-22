{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

-- | Example 31: Random number generation — normal distribution.
--
-- Generates a 2x3 tensor of random floats from a standard normal
-- distribution (mean = 0.0, std = 1.0).
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-rng-normal

module Main where

import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

import HHLO.Core.Types
import HHLO.EDSL.Ops
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
    putStrLn "=== Example 31: RNG Normal ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    let modu = moduleFromBuilder @'[2,3] @'F32 "main"
            []
            $ do
                r <- rngNormal
                return r

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    putStrLn "\nAttempting to compile and execute..."
    exec <- compile api client (render modu)
    [bufR] <- execute api exec []
    result <- fromDeviceF32 api bufR 6
    putStrLn $ "Result (2x3 normal floats): " ++ show (V.toList result)

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
