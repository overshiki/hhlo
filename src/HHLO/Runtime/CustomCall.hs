{-# LANGUAGE ForeignFunctionInterface #-}

-- | Runtime support for XLA custom-call symbol loading.
--
-- Custom-call kernels live in separate shared libraries (e.g. @libfoo.so@).
-- Before compiling or executing any HHLO module that references a custom
-- target, the library must be loaded and its target registered with the
-- runtime.
--
-- For CPU plugins, 'loadCustomCallLibrary' promotes symbols to the global
-- namespace so that XLA can resolve them via @dlsym(RTLD_DEFAULT, ...)@.
--
-- For GPU plugins, 'registerGpuCustomCall' uses the PJRT GPU custom-call
-- extension to register the target directly with the PJRT CUDA plugin.
--
-- Typical usage in application code:
--
-- @
-- main = withGPU $ \\sess -> do
--     registerGpuCustomCall (sessionApi sess) "lib/libmyplugin.so" "my_kernel"
--     let modu = buildMyModule   -- uses 'customCall1' inside
--     compiled <- compile sess modu
--     ...
-- @
module HHLO.Runtime.CustomCall
    ( loadCustomCallLibrary
    , registerGpuCustomCall
    ) where

import Data.Bits ((.|.))
import Foreign.C.String (CString, withCString, peekCString)
import Foreign.C.Types (CInt(..))
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, nullPtr)
import Foreign.Storable (peek)
import System.IO.Error (ioeSetLocation, mkIOError, userErrorType)

import HHLO.Runtime.PJRT.Types (PJRTApi(..))
import HHLO.Runtime.PJRT.FFI (c_pjrtRegisterGpuCustomCall)

-- RTLD_NOW   = 0x00002
-- RTLD_GLOBAL = 0x00100
-- Combined   = 0x00102 = 258
flagRTLD_NOW, flagRTLD_GLOBAL, flagCombined :: CInt
flagRTLD_NOW    = 2
flagRTLD_GLOBAL = 256
flagCombined    = flagRTLD_NOW .|. flagRTLD_GLOBAL

-- | Open a dynamic library and promote its symbols to the global namespace.
--
-- This is a thin wrapper around @dlopen(path, RTLD_NOW | RTLD_GLOBAL)@.
-- It is sufficient for CPU custom calls, where XLA resolves symbols via
-- @dlsym(RTLD_DEFAULT, ...)@.
--
-- For GPU custom calls, use 'registerGpuCustomCall' instead.
loadCustomCallLibrary :: FilePath -> IO ()
loadCustomCallLibrary path = do
    handle <- withCString path $ \cstr ->
        c_dlopen cstr flagCombined
    if handle /= nullPtr
        then return ()
        else ioError $ ioeSetLocation
             (mkIOError userErrorType
                 ("cannot load custom-call library: " ++ path)
                 Nothing Nothing)
             "HHLO.Runtime.CustomCall.loadCustomCallLibrary"

foreign import ccall "dlopen"
    c_dlopen :: CString -> CInt -> IO (Ptr ())

-- | Register a GPU custom-call target with the PJRT CUDA plugin.
--
-- This function:
-- 1. Opens the shared library at @libPath@.
-- 2. Looks up the symbol @targetName@.
-- 3. Registers it with the PJRT GPU custom-call extension.
--
-- The library handle is intentionally kept open for the process lifetime.
--
-- Must be called before 'compile'.
registerGpuCustomCall :: PJRTApi -> FilePath -> String -> IO ()
registerGpuCustomCall (PJRTApi apiPtr) libPath targetName =
    withCString libPath $ \cLibPath ->
    withCString targetName $ \cTargetName ->
    alloca $ \errMsgPtr -> do
        rc <- c_pjrtRegisterGpuCustomCall apiPtr cLibPath cTargetName errMsgPtr
        if rc == 0
            then return ()
            else do
                errMsg <- peek errMsgPtr >>= peekCString
                ioError $ ioeSetLocation
                    (mkIOError userErrorType
                        ("registerGpuCustomCall failed: " ++ errMsg)
                        Nothing Nothing)
                    "HHLO.Runtime.CustomCall.registerGpuCustomCall"
