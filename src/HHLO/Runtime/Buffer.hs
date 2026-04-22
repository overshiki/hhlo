{-# LANGUAGE ScopedTypeVariables #-}

module HHLO.Runtime.Buffer
    ( toDevice
    , toDeviceOn
    , fromDevice
    , fromDeviceAsync
    , toDeviceF32
    , fromDeviceF32
    -- * Buffer metadata queries
    , bufferDimensions
    , bufferElementType
    , bufferOnDeviceSize
    ) where

import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import Foreign.C
import qualified Foreign.Concurrent as Conc (newForeignPtr)
import Foreign.ForeignPtr (newForeignPtr)
import GHC.ForeignPtr (unsafeForeignPtrToPtr)
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
-- The returned buffer is managed by a 'ForeignPtr' finalizer that calls
-- 'PJRT_Buffer_Destroy' when the value is garbage-collected.
toDevice :: Storable a
         => PJRTApi -> PJRTClient -> Vector a -> [Int64] -> CInt -> IO PJRTBuffer
toDevice api client vec dims dtype =
    V.unsafeWith vec $ \ptr -> do
        withArrayLen (map fromIntegral dims :: [Int64]) $ \n dimArr -> do
            alloca $ \bufPtrPtr -> do
                checkError (unApi api) $ do
                    c_pjrtBufferFromHost (unApi api) (unClient client)
                        (castPtr ptr) dtype dimArr (fromIntegral n) bufPtrPtr
                rawPtr <- peek bufPtrPtr
                fp <- Conc.newForeignPtr rawPtr $ do
                    _ <- c_pjrtBufferDestroy (unApi api) rawPtr
                    return ()
                return $ PJRTBuffer fp

-- | Create a PJRT buffer on a specific device from a host 'Vector'.
toDeviceOn :: Storable a
           => PJRTApi -> PJRTClient -> PJRTDevice -> Vector a -> [Int64] -> CInt -> IO PJRTBuffer
toDeviceOn api client dev vec dims dtype =
    V.unsafeWith vec $ \ptr -> do
        withArrayLen (map fromIntegral dims :: [Int64]) $ \n dimArr -> do
            alloca $ \bufPtrPtr -> do
                checkError (unApi api) $ do
                    c_pjrtBufferFromHostOnDevice (unApi api) (unClient client) (unDevice dev)
                        (castPtr ptr) dtype dimArr (fromIntegral n) bufPtrPtr
                rawPtr <- peek bufPtrPtr
                fp <- Conc.newForeignPtr rawPtr $ do
                    _ <- c_pjrtBufferDestroy (unApi api) rawPtr
                    return ()
                return $ PJRTBuffer fp

-- | Convenience: create an F32 buffer from a Float vector.
toDeviceF32 :: PJRTApi -> PJRTClient -> Vector Float -> [Int64] -> IO PJRTBuffer
toDeviceF32 api client vec dims = toDevice api client vec dims bufferTypeF32

-- | Copy a PJRT buffer back to a host 'Vector'.
-- The caller must know the expected number of elements.
-- This is synchronous: it blocks until the data is ready on the host.
fromDevice :: forall a. Storable a => PJRTApi -> PJRTBuffer -> Int -> IO (Vector a)
fromDevice api buf numElems = do
    let totalBytes = numElems * sizeOf (undefined :: a)
    dstPtr <- mallocBytes totalBytes
    checkError (unApi api) $ do
        c_pjrtBufferToHost (unApi api) (unBuf buf)
            (castPtr dstPtr) (fromIntegral totalBytes) nullPtr
    fptr <- newForeignPtr finalizerFree dstPtr
    return $ V.unsafeFromForeignPtr0 fptr numElems

-- | Initiate an asynchronous device-to-host copy into a caller-provided
-- host buffer.  Returns a 'PJRTEvent' that signals when 'dst' is safe to
-- read.  The caller is responsible for allocating 'dst', awaiting the
-- event, and destroying the event.
fromDeviceAsync :: PJRTApi -> PJRTBuffer -> Ptr () -> Int -> IO (Ptr PJRTEvent)
fromDeviceAsync api buf dstPtr totalBytes =
    alloca $ \eventPtrPtr -> do
        checkError (unApi api) $ do
            c_pjrtBufferToHostAsync (unApi api) (unBuf buf)
                (castPtr dstPtr) (fromIntegral totalBytes) eventPtrPtr
        peek eventPtrPtr

-- | Convenience: read an F32 buffer back as a Float vector.
fromDeviceF32 :: PJRTApi -> PJRTBuffer -> Int -> IO (Vector Float)
fromDeviceF32 = fromDevice

-- | Query the dimensions (shape) of a device buffer.
-- Returns a list like @[batch, height, width, channels]@.
bufferDimensions :: PJRTApi -> PJRTBuffer -> IO [Int64]
bufferDimensions api buf = do
    alloca $ \dimsPtrPtr -> do
        alloca $ \numDimsPtr -> do
            checkError (unApi api) $ do
                c_pjrtBufferDimensions (unApi api) (unBuf buf) dimsPtrPtr numDimsPtr
            numDims <- peek numDimsPtr
            dimsPtr <- peek dimsPtrPtr
            peekArray (fromIntegral numDims) dimsPtr

-- | Query the element type of a device buffer.
-- Returns the PJRT_Buffer_Type enum value (e.g. 11 for F32).
bufferElementType :: PJRTApi -> PJRTBuffer -> IO CInt
bufferElementType api buf =
    alloca $ \typePtr -> do
        checkError (unApi api) $ do
            c_pjrtBufferElementType (unApi api) (unBuf buf) typePtr
        peek typePtr

-- | Query the on-device size of a buffer in bytes.
bufferOnDeviceSize :: PJRTApi -> PJRTBuffer -> IO Int
bufferOnDeviceSize api buf =
    alloca $ \sizePtr -> do
        checkError (unApi api) $ do
            c_pjrtBufferOnDeviceSize (unApi api) (unBuf buf) sizePtr
        fromIntegral <$> peek sizePtr

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p

unDevice :: PJRTDevice -> Ptr PJRTDevice
unDevice (PJRTDevice p) = p

unBuf :: PJRTBuffer -> Ptr PJRTBuffer
unBuf (PJRTBuffer fp) = unsafeForeignPtrToPtr fp
