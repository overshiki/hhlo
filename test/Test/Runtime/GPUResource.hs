{-# LANGUAGE ForeignFunctionInterface #-}

module Test.Runtime.GPUResource
    ( GPUResource(..)
    , acquireGPU
    , releaseGPU
    ) where

import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.Device

data GPUResource = GPUResource
    { resApi    :: !PJRTApi
    , resClient :: !PJRTClient
    , resDevice :: !PJRTDevice
    }

acquireGPU :: IO GPUResource
acquireGPU = do
    api <- withCString "deps/pjrt/libpjrt_cuda.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr
    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr
    mDev <- defaultGPUDevice api client
    dev <- maybe (error "No GPU found") return mDev
    return $ GPUResource api client dev
  where
    unApi (PJRTApi p) = p

releaseGPU :: GPUResource -> IO ()
releaseGPU res = do
    let api = resApi res
        client = resClient res
    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
