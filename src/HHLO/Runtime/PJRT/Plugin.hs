{-# LANGUAGE ForeignFunctionInterface #-}

module HHLO.Runtime.PJRT.Plugin
    ( withPJRT
    , withPJRTCPU
    , withPJRTGPU
    ) where

import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error

-- | Load a PJRT plugin from the given file path, create a client,
-- run the action, then destroy the client.
--
-- The plugin handle is intentionally leaked (not dlclosed) for program
-- lifetime simplicity; the OS cleans it up on process exit.
withPJRT :: FilePath -> (PJRTApi -> PJRTClient -> IO a) -> IO a
withPJRT pluginPath action = do
    api <- withCString pluginPath $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    result <- action api client

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
    return result

-- | Convenience wrapper for the CPU PJRT plugin.
withPJRTCPU :: (PJRTApi -> PJRTClient -> IO a) -> IO a
withPJRTCPU = withPJRT "deps/pjrt/libpjrt_cpu.so"

-- | Convenience wrapper for the CUDA PJRT plugin.
withPJRTGPU :: (PJRTApi -> PJRTClient -> IO a) -> IO a
withPJRTGPU = withPJRT "deps/pjrt/libpjrt_cuda.so"

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p
