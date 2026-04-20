{-# LANGUAGE ScopedTypeVariables #-}

module HHLO.Runtime.Buffer
    ( toDevice
    , fromDevice
    , toDeviceF32
    , fromDeviceF32
    , destroyBuffer
    ) where

import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import Foreign.C
import Foreign.C.Types (CInt)
import Foreign.ForeignPtr
import Data.Int (Int64)
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error

-- | Create a PJRT buffer from a host 'Vector'.
-- The vector data must be contiguous (which 'Vector' guarantees).
toDevice :: Storable a
         => PJRTApi -> PJRTClient -> Vector a -> [Int64] -> CInt -> IO PJRTBuffer
toDevice api client vec dims dtype =
    V.unsafeWith vec $ \ptr -> do
        withArrayLen (map fromIntegral dims) $ \n dimArr -> do
            alloca $ \bufPtrPtr -> do
                checkError (unApi api) $ do
                    c_pjrtBufferFromHost (unApi api) (unClient client)
                        (castPtr ptr) dtype dimArr (fromIntegral n) bufPtrPtr
                bufPtr <- peek bufPtrPtr
                return $ PJRTBuffer bufPtr

-- | Convenience: create an F32 buffer from a Float vector.
toDeviceF32 :: PJRTApi -> PJRTClient -> Vector Float -> [Int64] -> IO PJRTBuffer
toDeviceF32 api client vec dims = toDevice api client vec dims 11  -- PJRT_Buffer_Type_F32 = 11

-- | Copy a PJRT buffer back to a host 'Vector'.
-- The caller must know the expected number of elements.
fromDevice :: forall a. Storable a => PJRTApi -> PJRTBuffer -> Int -> IO (Vector a)
fromDevice api buf numElems = do
    let totalBytes = numElems * sizeOf (undefined :: a)
    dstPtr <- mallocBytes totalBytes
    checkError (unApi api) $ do
        c_pjrtBufferToHost (unApi api) (unBuf buf)
            (castPtr dstPtr) (fromIntegral totalBytes) nullPtr
    fptr <- newForeignPtr finalizerFree dstPtr
    return $ V.unsafeFromForeignPtr0 fptr numElems

-- | Convenience: read an F32 buffer back as a Float vector.
fromDeviceF32 :: PJRTApi -> PJRTBuffer -> Int -> IO (Vector Float)
fromDeviceF32 = fromDevice

-- | Destroy a device buffer, releasing its memory.
destroyBuffer :: PJRTApi -> PJRTBuffer -> IO ()
destroyBuffer api buf = do
    checkError (unApi api) $ c_pjrtBufferDestroy (unApi api) (unBuf buf)

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p

unBuf :: PJRTBuffer -> Ptr PJRTBuffer
unBuf (PJRTBuffer p) = p
