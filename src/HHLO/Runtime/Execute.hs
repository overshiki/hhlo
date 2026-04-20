module HHLO.Runtime.Execute
    ( execute
    , executeAsync
    ) where

import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import Control.Exception (throwIO)
import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error (PJRTException(..), withErrorMessage)

-- | Execute a compiled program synchronously (blocking).
-- Returns the list of output buffers.
execute :: PJRTApi -> PJRTExecutable -> [PJRTBuffer] -> IO [PJRTBuffer]
execute api exec buffers = do
    -- Prepare output slots (assume at most 8 outputs for now)
    let maxOutputs = 8
    withArrayLen (map unBuffer buffers) $ \n bufArr -> do
        allocaArray maxOutputs $ \outArr -> do
            pokeArray outArr (replicate maxOutputs nullPtr)
            alloca $ \numOutPtr -> do
                err <- c_pjrtExecute (unApi api) (unExec exec)
                        (fromIntegral n) bufArr
                        (fromIntegral maxOutputs) outArr numOutPtr
                if err == nullPtr
                    then do
                        numOut <- peek numOutPtr
                        outPtrs <- peekArray (fromIntegral numOut) outArr
                        return $ map PJRTBuffer outPtrs
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
unExec (PJRTExecutable p) = p

unBuffer :: PJRTBuffer -> Ptr PJRTBuffer
unBuffer (PJRTBuffer p) = p
