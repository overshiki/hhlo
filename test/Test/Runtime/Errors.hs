{-# LANGUAGE OverloadedStrings #-}

module Test.Runtime.Errors where

import qualified Data.Text as T
import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.Compile
import Test.Utils

tests :: TestTree
tests = testGroup "Runtime.Errors"
    [ assertThrowsPJRT "malformed mlir" $ withPJRTCPU $ \api client -> do
        _ <- compile api client "not valid mlir"
        return ()
    ]
