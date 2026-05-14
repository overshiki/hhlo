module HHLO.Runtime.Execute
    ( execute
    , executeOn
    , executeAsync
    , executeReplicas
    ) where

import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import qualified Foreign.Concurrent as Conc (newForeignPtr)

import Control.Concurrent.Async (mapConcurrently)
import Control.Exception (throwIO)
import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error (PJRTException(..), withErrorMessage)

-- | Execute a compiled program synchronously (blocking).
-- Returns the list of output buffers.
execute :: PJRTApi -> PJRTExecutable -> [PJRTBuffer] -> IO [PJRTBuffer]
execute api exec buffers = withExecPtr exec $ \execPtr -> do
    -- Query the executable's actual output count instead of hardcoding.
    numOutputs <- alloca $ \numOutPtr -> do
        err <- c_pjrtExecutableNumOutputs (unApi api) execPtr numOutPtr
        if err == nullPtr
            then peek numOutPtr
            else do
                withErrorMessage (unApi api) err >>= throwIO . PJRTException
    withBufferPtrs buffers $ \bufPtrs ->
        withArrayLen bufPtrs $ \n bufArr -> do
            allocaArray (fromIntegral numOutputs) $ \outArr -> do
                pokeArray outArr (replicate (fromIntegral numOutputs) nullPtr)
                alloca $ \numOutPtr -> do
                    err <- c_pjrtExecute (unApi api) execPtr
                            (fromIntegral n) bufArr
                            numOutputs outArr numOutPtr
                    if err == nullPtr
                        then do
                            actualNumOut <- peek numOutPtr
                            outPtrs <- peekArray (fromIntegral actualNumOut) outArr
                            mapM (wrapBuffer api) outPtrs
                        else do
                            withErrorMessage (unApi api) err >>= throwIO . PJRTException

-- | Execute on a specific device.
executeOn :: PJRTApi -> PJRTExecutable -> PJRTDevice -> [PJRTBuffer] -> IO [PJRTBuffer]
executeOn api exec dev buffers = withExecPtr exec $ \execPtr -> do
    numOutputs <- alloca $ \numOutPtr -> do
        err <- c_pjrtExecutableNumOutputs (unApi api) execPtr numOutPtr
        if err == nullPtr
            then peek numOutPtr
            else do
                withErrorMessage (unApi api) err >>= throwIO . PJRTException
    withBufferPtrs buffers $ \bufPtrs ->
        withArrayLen bufPtrs $ \n bufArr -> do
            allocaArray (fromIntegral numOutputs) $ \outArr -> do
                pokeArray outArr (replicate (fromIntegral numOutputs) nullPtr)
                alloca $ \numOutPtr -> do
                    err <- c_pjrtExecuteOnDevice (unApi api) execPtr
                            (fromIntegral n) bufArr (unDevice dev)
                            numOutputs outArr numOutPtr
                    if err == nullPtr
                        then do
                            actualNumOut <- peek numOutPtr
                            outPtrs <- peekArray (fromIntegral actualNumOut) outArr
                            mapM (wrapBuffer api) outPtrs
                        else do
                            withErrorMessage (unApi api) err >>= throwIO . PJRTException

-- | Execute asynchronously. Returns immediately with output buffers
-- that may not yet contain valid data. The caller must synchronize
-- via the buffer's ready event or by copying to host.
executeAsync :: PJRTApi -> PJRTExecutable -> [PJRTBuffer] -> IO [PJRTBuffer]
executeAsync = execute  -- For now, same implementation; can be optimized later

-- | Execute the same compiled program concurrently on multiple devices.
-- Each device receives its own input buffers; outputs are gathered
-- in the same order as the input device list.
--
-- This is the recommended API for multi-GPU inference scaling.
-- It launches independent executions via Haskell's async threads,
-- which the underlying PJRT CUDA plugin schedules onto each GPU.
executeReplicas :: PJRTApi -> PJRTExecutable -> [(PJRTDevice, [PJRTBuffer])] -> IO [[PJRTBuffer]]
executeReplicas api exec deviceArgs =
    mapConcurrently (uncurry (executeOn api exec)) deviceArgs

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unDevice :: PJRTDevice -> Ptr PJRTDevice
unDevice (PJRTDevice p) = p

-- | Wrap a raw PJRT buffer pointer in a 'ForeignPtr' with a finalizer.
wrapBuffer :: PJRTApi -> Ptr PJRTBuffer -> IO PJRTBuffer
wrapBuffer api rawPtr = do
    fp <- Conc.newForeignPtr rawPtr $ do
        _ <- c_pjrtBufferDestroy (unApi api) rawPtr
        return ()
    return $ PJRTBuffer fp
