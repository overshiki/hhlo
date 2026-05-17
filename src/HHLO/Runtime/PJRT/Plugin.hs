{-# LANGUAGE ForeignFunctionInterface #-}

module HHLO.Runtime.PJRT.Plugin
    ( withPJRT
    , withPJRTCPU
    , withPJRTGPU
    , getPluginPath
    ) where

import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

import Data.Char (toUpper)
import System.Directory (doesFileExist)
import System.Environment (lookupEnv)

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.PJRT.Registry (registerApi, unregisterApi)

-- | Load a PJRT plugin from the given file path, create a client,
-- run the action, then destroy the client.
--
-- The plugin handle is intentionally leaked (not dlclosed) for program
-- lifetime simplicity; the OS cleans it up on process exit.
withPJRT :: FilePath -> (PJRTApi -> PJRTClient -> IO a) -> IO a
withPJRT pluginPath action = do
    api <- withCString pluginPath $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    registerApi api

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    result <- action api client

    unregisterApi api
    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
    return result

-- | Convenience wrapper for the CPU PJRT plugin.
--
-- The plugin path is resolved via 'getPluginPath'.
withPJRTCPU :: (PJRTApi -> PJRTClient -> IO a) -> IO a
withPJRTCPU action = do
    path <- getPluginPath "cpu" "libpjrt_cpu.so"
    withPJRT path action

-- | Convenience wrapper for the CUDA PJRT plugin.
--
-- The plugin path is resolved via 'getPluginPath'.
withPJRTGPU :: (PJRTApi -> PJRTClient -> IO a) -> IO a
withPJRTGPU action = do
    path <- getPluginPath "gpu" "libpjrt_cuda.so"
    withPJRT path action

-- | Search for a PJRT plugin.
--
-- Priority:
--   1. @HHLO_PJRT_<PLATFORM>_PLUGIN@ environment variable
--   2. @deps/pjrt/<defaultName>@ (downloaded by @pjrt_script.sh@)
--   3. Runtime error with instructions
getPluginPath :: String -> FilePath -> IO FilePath
getPluginPath platform defaultName = do
    mEnv <- lookupEnv ("HHLO_PJRT_" ++ map toUpper platform ++ "_PLUGIN")
    case mEnv of
        Just p  -> return p
        Nothing -> do
            let defaultPath = "deps/pjrt/" ++ defaultName
            exists <- doesFileExist defaultPath
            if exists
                then return defaultPath
                else error $ unlines
                    [ "PJRT " ++ platform ++ " plugin not found at: " ++ defaultPath
                    , ""
                    , "To fix this, either:"
                    , "  1. Run the download script:"
                    , "       ./pjrt_script.sh"
                    , "  2. Set the environment variable:"
                    , "       export HHLO_PJRT_" ++ map toUpper platform ++ "_PLUGIN=/path/to/" ++ defaultName
                    ]

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p
