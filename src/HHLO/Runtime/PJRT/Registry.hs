{-# LANGUAGE Unsafe #-}

-- | Process-wide registry tracking which 'PJRTApi' pointers are currently
-- alive (i.e. have an open session).  Buffer finalizers use this to avoid
-- calling 'PJRT_Buffer_Destroy' after the client has been torn down,
-- which would segfault because the buffer's internal state references
-- client-owned resources (CUDA context, StreamExecutor, etc.) that are
-- destroyed when the client closes.
--
-- This module is marked @Unsafe@ because it uses 'unsafePerformIO' for
-- the top-level 'IORef'.  It is only imported by internal hhlo modules.
module HHLO.Runtime.PJRT.Registry
    ( registerApi
    , unregisterApi
    , isApiAlive
    ) where

import Data.IORef (IORef, atomicModifyIORef', newIORef, readIORef)
import qualified Data.Map.Strict as Map
import Foreign.Ptr (Ptr)
import System.IO.Unsafe (unsafePerformIO)

import HHLO.Runtime.PJRT.Types (PJRTApi(..))

-- | Global registry of alive API pointers.
apiRegistry :: IORef (Map.Map (Ptr PJRTApi) ())
apiRegistry = unsafePerformIO $ newIORef Map.empty
{-# NOINLINE apiRegistry #-}

-- | Register a 'PJRTApi' as alive.  Called by 'withPJRT' when a session
-- opens.
registerApi :: PJRTApi -> IO ()
registerApi api =
    atomicModifyIORef' apiRegistry $ \m ->
        (Map.insert (unApi api) () m, ())

-- | Unregister a 'PJRTApi'.  Called by 'withPJRT' when a session closes.
-- After this, buffer finalizers that reference this API will skip their
-- destroy call.
unregisterApi :: PJRTApi -> IO ()
unregisterApi api =
    atomicModifyIORef' apiRegistry $ \m ->
        (Map.delete (unApi api) m, ())

-- | Check whether a raw 'PJRTApi' pointer is currently registered as
-- alive.
isApiAlive :: Ptr PJRTApi -> IO Bool
isApiAlive p =
    Map.member p <$> readIORef apiRegistry

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p
