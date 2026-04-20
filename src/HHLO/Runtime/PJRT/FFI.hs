{-# LANGUAGE ForeignFunctionInterface #-}

module HHLO.Runtime.PJRT.FFI where

import Foreign.C
import Foreign.Ptr
import Data.Int (Int64)

import HHLO.Runtime.PJRT.Types

-- ---------------------------------------------------------------------------
-- Plugin loading
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_load_plugin"
    c_pjrtLoadPlugin :: CString -> Ptr (Ptr PJRTApi) -> IO (Ptr PJRTError)

-- ---------------------------------------------------------------------------
-- Client
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_create_client"
    c_pjrtCreateClient :: Ptr PJRTApi -> Ptr (Ptr PJRTClient) -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_client_destroy"
    c_pjrtClientDestroy :: Ptr PJRTApi -> Ptr PJRTClient -> IO (Ptr PJRTError)

-- ---------------------------------------------------------------------------
-- Compilation
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_compile"
    c_pjrtCompile :: Ptr PJRTApi
                  -> Ptr PJRTClient
                  -> CString
                  -> CSize
                  -> Ptr (Ptr PJRTExecutable)
                  -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_loaded_executable_destroy"
    c_pjrtLoadedExecutableDestroy :: Ptr PJRTApi -> Ptr PJRTExecutable -> IO (Ptr PJRTError)

-- ---------------------------------------------------------------------------
-- Execution
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_execute"
    c_pjrtExecute :: Ptr PJRTApi
                  -> Ptr PJRTExecutable
                  -> CSize               -- num_args
                  -> Ptr (Ptr PJRTBuffer) -- args
                  -> CSize               -- max_outputs
                  -> Ptr (Ptr PJRTBuffer) -- out_outputs
                  -> Ptr CSize            -- out_num_outputs
                  -> IO (Ptr PJRTError)

-- ---------------------------------------------------------------------------
-- Buffers
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_from_host"
    c_pjrtBufferFromHost :: Ptr PJRTApi
                         -> Ptr PJRTClient
                         -> Ptr ()              -- data
                         -> CInt                -- dtype (PJRT_Buffer_Type)
                         -> Ptr Int64           -- dims
                         -> CSize               -- num_dims
                         -> Ptr (Ptr PJRTBuffer)
                         -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_to_host"
    c_pjrtBufferToHost :: Ptr PJRTApi
                       -> Ptr PJRTBuffer
                       -> Ptr ()               -- dst
                       -> CSize                -- dst_size
                       -> Ptr (Ptr PJRTEvent)  -- out_event
                       -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_destroy"
    c_pjrtBufferDestroy :: Ptr PJRTApi -> Ptr PJRTBuffer -> IO (Ptr PJRTError)

-- ---------------------------------------------------------------------------
-- Events
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_event_await"
    c_pjrtEventAwait :: Ptr PJRTApi -> Ptr PJRTEvent -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_event_destroy"
    c_pjrtEventDestroy :: Ptr PJRTApi -> Ptr PJRTEvent -> IO (Ptr PJRTError)

-- ---------------------------------------------------------------------------
-- Errors
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_error_message"
    c_pjrtErrorMessage :: Ptr PJRTApi
                       -> Ptr PJRTError
                       -> Ptr CString
                       -> Ptr CSize
                       -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_error_destroy"
    c_pjrtErrorDestroy :: Ptr PJRTApi -> Ptr PJRTError -> IO (Ptr PJRTError)
