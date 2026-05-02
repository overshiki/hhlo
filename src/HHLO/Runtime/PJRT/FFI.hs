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
-- Device enumeration
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_client_addressable_device_count"
    c_pjrtClientAddressableDeviceCount :: Ptr PJRTApi -> Ptr PJRTClient -> Ptr CSize -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_client_addressable_device"
    c_pjrtClientAddressableDevice :: Ptr PJRTApi -> Ptr PJRTClient -> CSize -> Ptr (Ptr PJRTDevice) -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_device_id"
    c_pjrtDeviceId :: Ptr PJRTApi -> Ptr PJRTDevice -> Ptr CInt -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_device_kind"
    c_pjrtDeviceKind :: Ptr PJRTApi -> Ptr PJRTDevice -> Ptr CString -> Ptr CSize -> IO (Ptr PJRTError)

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

foreign import ccall "pjrt_shim.h hhlo_pjrt_compile_with_options"
    c_pjrtCompileWithOptions :: Ptr PJRTApi
                             -> Ptr PJRTClient
                             -> CString
                             -> CSize
                             -> CInt          -- num_replicas
                             -> Ptr (Ptr PJRTExecutable)
                             -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_loaded_executable_destroy"
    c_pjrtLoadedExecutableDestroy :: Ptr PJRTApi -> Ptr PJRTExecutable -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_executable_num_outputs"
    c_pjrtExecutableNumOutputs :: Ptr PJRTApi -> Ptr PJRTExecutable -> Ptr CSize -> IO (Ptr PJRTError)

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

foreign import ccall "pjrt_shim.h hhlo_pjrt_execute_on_device"
    c_pjrtExecuteOnDevice :: Ptr PJRTApi
                          -> Ptr PJRTExecutable
                          -> CSize                -- num_args
                          -> Ptr (Ptr PJRTBuffer) -- args
                          -> Ptr PJRTDevice       -- execute_device
                          -> CSize                -- max_outputs
                          -> Ptr (Ptr PJRTBuffer) -- out_outputs
                          -> Ptr CSize             -- out_num_outputs
                          -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_execute_multi"
    c_pjrtExecuteMulti :: Ptr PJRTApi
                       -> Ptr PJRTExecutable
                       -> CSize                   -- num_devices
                       -> CSize                   -- num_args
                       -> Ptr (Ptr (Ptr PJRTBuffer)) -- args_in (device × arg)
                       -> CSize                   -- max_outputs
                       -> Ptr (Ptr (Ptr PJRTBuffer)) -- out_outputs (device × output)
                       -> Ptr CSize                -- out_num_outputs_per_device
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

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_from_host_on_device"
    c_pjrtBufferFromHostOnDevice :: Ptr PJRTApi
                                 -> Ptr PJRTClient
                                 -> Ptr PJRTDevice          -- device
                                 -> Ptr ()                  -- data
                                 -> CInt                    -- dtype
                                 -> Ptr Int64               -- dims
                                 -> CSize                   -- num_dims
                                 -> Ptr (Ptr PJRTBuffer)
                                 -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_to_host"
    c_pjrtBufferToHost :: Ptr PJRTApi
                       -> Ptr PJRTBuffer
                       -> Ptr ()               -- dst
                       -> CSize                -- dst_size
                       -> Ptr (Ptr PJRTEvent)  -- out_event
                       -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_to_host_async"
    c_pjrtBufferToHostAsync :: Ptr PJRTApi
                            -> Ptr PJRTBuffer
                            -> Ptr ()               -- dst
                            -> CSize                -- dst_size
                            -> Ptr (Ptr PJRTEvent)  -- out_event
                            -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_destroy"
    c_pjrtBufferDestroy :: Ptr PJRTApi -> Ptr PJRTBuffer -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_dimensions"
    c_pjrtBufferDimensions :: Ptr PJRTApi -> Ptr PJRTBuffer -> Ptr (Ptr Int64) -> Ptr CSize -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_element_type"
    c_pjrtBufferElementType :: Ptr PJRTApi -> Ptr PJRTBuffer -> Ptr CInt -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_on_device_size"
    c_pjrtBufferOnDeviceSize :: Ptr PJRTApi -> Ptr PJRTBuffer -> Ptr CSize -> IO (Ptr PJRTError)

-- ---------------------------------------------------------------------------
-- Events
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_buffer_ready_event"
    c_pjrtBufferReadyEvent :: Ptr PJRTApi
                           -> Ptr PJRTBuffer
                           -> Ptr (Ptr PJRTEvent)
                           -> IO (Ptr PJRTError)

foreign import ccall "pjrt_shim.h hhlo_pjrt_event_is_ready"
    c_pjrtEventIsReady :: Ptr PJRTApi -> Ptr PJRTEvent -> Ptr CInt -> IO (Ptr PJRTError)

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

-- ---------------------------------------------------------------------------
-- Custom calls
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_pjrt_register_gpu_custom_call"
    c_pjrtRegisterGpuCustomCall :: Ptr PJRTApi
                                -> CString            -- lib_path
                                -> CString            -- function_name
                                -> Ptr CString        -- out_error_msg
                                -> IO CInt
