{-# LANGUAGE ForeignFunctionInterface #-}

module HHLO.Runtime.Device
    ( addressableDevices
    , deviceId
    , deviceKind
    , defaultGPUDevice
    ) where

import Data.Char (toLower)
import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray)
import Foreign.Ptr
import Foreign.Storable (peek)

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error

-- | List all devices that this client can directly execute on.
addressableDevices :: PJRTApi -> PJRTClient -> IO [PJRTDevice]
addressableDevices api client = do
    count <- alloca $ \countPtr -> do
        checkError (unApi api) $
            c_pjrtClientAddressableDeviceCount (unApi api) (unClient client) countPtr
        peek countPtr
    mapM (getDevice api client . fromIntegral) [0 .. count - 1]
  where
    getDevice :: PJRTApi -> PJRTClient -> Int -> IO PJRTDevice
    getDevice a c idx =
        alloca $ \devPtrPtr -> do
            checkError (unApi a) $
                c_pjrtClientAddressableDevice (unApi a) (unClient c) (fromIntegral idx) devPtrPtr
            PJRTDevice <$> peek devPtrPtr

-- | Get the stable ID of a device.
deviceId :: PJRTApi -> PJRTDevice -> IO Int
deviceId api dev =
    alloca $ \idPtr -> do
        checkError (unApi api) $
            c_pjrtDeviceId (unApi api) (unDevice dev) idPtr
        fromIntegral <$> peek idPtr

-- | Get the platform kind of a device, e.g. @"CPU"@ or @"CUDA"@.
deviceKind :: PJRTApi -> PJRTDevice -> IO String
deviceKind api dev =
    alloca $ \kindPtr -> do
        alloca $ \lenPtr -> do
            checkError (unApi api) $
                c_pjrtDeviceKind (unApi api) (unDevice dev) kindPtr lenPtr
            cstr <- peek kindPtr
            len  <- peek lenPtr
            peekCStringLen (cstr, fromIntegral len)

-- | Return the first non-CPU device, or 'Nothing' if only CPUs exist.
-- For NVIDIA GPU plugins the kind string is the GPU name (e.g.
-- @"NVIDIA GeForce RTX 5090"@), so we detect CPU by checking for the
-- substring @"cpu"@ (case-insensitive).
defaultGPUDevice :: PJRTApi -> PJRTClient -> IO (Maybe PJRTDevice)
defaultGPUDevice api client = do
    devs <- addressableDevices api client
    let findGPU [] = return Nothing
        findGPU (d:ds) = do
            kindStr <- deviceKind api d
            if map toLower kindStr == "cpu"
                then findGPU ds
                else return (Just d)
    findGPU devs

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p

unDevice :: PJRTDevice -> Ptr PJRTDevice
unDevice (PJRTDevice p) = p
