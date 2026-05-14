{-# LANGUAGE ScopedTypeVariables #-}

module HHLO.Runtime.Async
    ( executeAsync
    , bufferReady
    , awaitBuffers
    ) where

import Foreign.C
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.Execute (execute)

-- | Execute a loaded executable asynchronously.
--
-- This is a semantic alias for 'execute' — PJRT execution is already
-- asynchronous at the device level.  The returned 'PJRTBuffer's are
-- valid handles immediately, but their host-visible data is not safe
-- to read until the buffer has been synchronised (see 'awaitBuffers').
executeAsync :: PJRTApi -> PJRTExecutable -> [PJRTBuffer] -> IO [PJRTBuffer]
executeAsync = execute

-- | Return 'True' if the buffer's computation has completed and the
-- data is ready on device.
--
-- This is a non-blocking poll.
bufferReady :: PJRTApi -> PJRTBuffer -> IO Bool
bufferReady api buf = do
    -- Obtain a ready-event for the buffer
    eventPtr <- alloca $ \evPtr -> do
        withBufferPtr buf $ \bufPtr -> do
            checkError (unApi api) $
                c_pjrtBufferReadyEvent (unApi api) bufPtr evPtr
        peek evPtr
    -- Check if the event is already ready
    ready <- alloca $ \readyPtr -> do
        checkError (unApi api) $
            c_pjrtEventIsReady (unApi api) eventPtr readyPtr
        (0 /=) <$> peek readyPtr
    -- Destroy the event handle
    _ <- c_pjrtEventDestroy (unApi api) eventPtr
    return ready

-- | Block until all buffers are ready (i.e. their device-side
-- computations have completed).
awaitBuffers :: PJRTApi -> [PJRTBuffer] -> IO ()
awaitBuffers api buffers =
    mapM_ (awaitBuffer api) buffers
  where
    awaitBuffer :: PJRTApi -> PJRTBuffer -> IO ()
    awaitBuffer a b = do
        eventPtr <- alloca $ \evPtr -> do
            withBufferPtr b $ \bufPtr -> do
                checkError (unApi a) $
                    c_pjrtBufferReadyEvent (unApi a) bufPtr evPtr
            peek evPtr
        checkError (unApi a) $
            c_pjrtEventAwait (unApi a) eventPtr
        _ <- c_pjrtEventDestroy (unApi a) eventPtr
        return ()

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p
