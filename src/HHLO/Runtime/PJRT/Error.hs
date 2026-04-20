{-# LANGUAGE ForeignFunctionInterface #-}

module HHLO.Runtime.PJRT.Error
    ( PJRTException(..)
    , checkError
    , withErrorMessage
    ) where

import Control.Exception
import Foreign.C
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.Storable
import qualified Data.Text as T
import Data.Text.Encoding (decodeUtf8)
import qualified Data.ByteString as BS

import HHLO.Runtime.PJRT.FFI (c_pjrtErrorMessage, c_pjrtErrorDestroy)
import HHLO.Runtime.PJRT.Types (PJRTApi, PJRTError)

data PJRTException = PJRTException !String
    deriving (Eq, Show)

instance Exception PJRTException

-- | Check a returned PJRT error pointer. If non-null, extract the message,
-- destroy the error, and throw a 'PJRTException'.
checkError :: Ptr PJRTApi -> IO (Ptr PJRTError) -> IO ()
checkError api action = do
    err <- action
    if err == nullPtr
        then return ()
        else withErrorMessage api err >>= throwIO . PJRTException

-- | Extract the human-readable message from a PJRT error, then destroy it.
withErrorMessage :: Ptr PJRTApi -> Ptr PJRTError -> IO String
withErrorMessage api err = do
    alloca $ \msgPtrPtr -> do
        alloca $ \sizePtr -> do
            err2 <- c_pjrtErrorMessage api err msgPtrPtr sizePtr
            if err2 /= nullPtr
                then do
                    _ <- c_pjrtErrorDestroy api err2
                    _ <- c_pjrtErrorDestroy api err
                    return "PJRT error (failed to retrieve message)"
                else do
                    msgPtr <- peek msgPtrPtr
                    msgSize <- peek sizePtr
                    msgBs <- BS.packCStringLen (msgPtr, fromIntegral msgSize)
                    _ <- c_pjrtErrorDestroy api err
                    return $ T.unpack $ decodeUtf8 msgBs
