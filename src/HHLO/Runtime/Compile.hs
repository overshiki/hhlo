module HHLO.Runtime.Compile
    ( compile
    ) where

import Control.Exception (throwIO)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString as BS
import Foreign.Concurrent (newForeignPtr)
import GHC.ForeignPtr (unsafeForeignPtrToPtr)
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.Storable

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error

-- | Compile a StableHLO MLIR text program into a PJRT executable.
-- The returned executable is managed by a 'ForeignPtr' finalizer that calls
-- 'PJRT_LoadedExecutable_Destroy' when the value is garbage-collected.
compile :: PJRTApi -> PJRTClient -> T.Text -> IO PJRTExecutable
compile api client mlirText = do
    let utf8 = TE.encodeUtf8 mlirText
    alloca $ \execPtrPtr -> do
        err <- BS.useAsCStringLen utf8 $ \(cstr, len) -> do
            c_pjrtCompile (unApi api) (unClient client) cstr (fromIntegral len) execPtrPtr
        if err == nullPtr
            then do
                rawPtr <- peek execPtrPtr
                fp <- newForeignPtr rawPtr $ do
                    _ <- c_pjrtLoadedExecutableDestroy (unApi api) rawPtr
                    return ()
                return $ PJRTExecutable fp
            else do
                withErrorMessage (unApi api) err >>= throwIO . PJRTException

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p
