{-# LANGUAGE ForeignFunctionInterface #-}

-- | Runtime support for XLA custom-call symbol loading.
--
-- Custom-call kernels live in separate shared libraries (e.g. @libfoo.so@).
-- Before compiling or executing any HHLO module that references a custom
-- target, the host process must load the library with 'RTLD_GLOBAL' so that
-- XLA's internal @dlsym(RTLD_DEFAULT, ...)@ can resolve the symbol.
--
-- Typical usage in application code:
--
-- @
-- main = withGPU $ \\sess -> do
--     loadCustomCallLibrary "lib/libmyplugin.so"
--     let modu = buildMyModule   -- uses 'customCall1' inside
--     compiled <- compile sess modu
--     ...
-- @
module HHLO.Runtime.CustomCall
    ( loadCustomCallLibrary
    ) where

import Data.Bits ((.|.))
import Foreign.C.String (CString, withCString)
import Foreign.C.Types (CInt(..))
import Foreign.Ptr (Ptr, nullPtr)
import System.IO.Error (ioeSetLocation, mkIOError, doesNotExistErrorType)

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
-- If the file does not exist or cannot be loaded, an 'IOError' is thrown.
loadCustomCallLibrary :: FilePath -> IO ()
loadCustomCallLibrary path = do
    handle <- withCString path $ \cstr ->
        c_dlopen cstr flagCombined
    if handle /= nullPtr
        then return ()
        else ioError $ ioeSetLocation
             (mkIOError doesNotExistErrorType
                 ("cannot load custom-call library: " ++ path)
                 Nothing Nothing)
             "HHLO.Runtime.CustomCall.loadCustomCallLibrary"

foreign import ccall "dlopen"
    c_dlopen :: CString -> CInt -> IO (Ptr ())
