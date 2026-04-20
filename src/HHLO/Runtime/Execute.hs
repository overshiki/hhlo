module HHLO.Runtime.Execute
    ( execute
    , executeAsync
    ) where

import GHC.ForeignPtr (unsafeForeignPtrToPtr)
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import qualified Foreign.Concurrent as Conc (newForeignPtr)

import Control.Exception (throwIO)
import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error (PJRTException(..), withErrorMessage)

-- | Execute a compiled program synchronously (blocking).
-- Returns the list of output buffers.
execute :: PJRTApi -> PJRTExecutable -> [PJRTBuffer] -> IO [PJRTBuffer]
execute api exec buffers = do
    -- Query the executable's actual output count instead of hardcoding.
    numOutputs <- alloca $ \numOutPtr -> do
        err <- c_pjrtExecutableNumOutputs (unApi api) (unExec exec) numOutPtr
        if err == nullPtr
            then peek numOutPtr
            else do
                withErrorMessage (unApi api) err >>= throwIO . PJRTException
    withArrayLen (map unBuffer buffers) $ \n bufArr -> do
        allocaArray (fromIntegral numOutputs) $ \outArr -> do
            pokeArray outArr (replicate (fromIntegral numOutputs) nullPtr)
            alloca $ \numOutPtr -> do
                err <- c_pjrtExecute (unApi api) (unExec exec)
                        (fromIntegral n) bufArr
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

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unExec :: PJRTExecutable -> Ptr PJRTExecutable
unExec (PJRTExecutable fp) = unsafeForeignPtrToPtr fp

unBuffer :: PJRTBuffer -> Ptr PJRTBuffer
unBuffer (PJRTBuffer fp) = unsafeForeignPtrToPtr fp

-- | Wrap a raw PJRT buffer pointer in a 'ForeignPtr' with a finalizer.
wrapBuffer :: PJRTApi -> Ptr PJRTBuffer -> IO PJRTBuffer
wrapBuffer api rawPtr = do
    fp <- Conc.newForeignPtr rawPtr $ do
        _ <- c_pjrtBufferDestroy (unApi api) rawPtr
        return ()
    return $ PJRTBuffer fp
