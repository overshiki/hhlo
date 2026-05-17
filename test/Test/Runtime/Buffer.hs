{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.Buffer where

import Control.Concurrent (threadDelay)
import Data.IORef (newIORef, readIORef, writeIORef)
import qualified Foreign.Concurrent as Conc (addForeignPtrFinalizer)
import qualified Data.Vector.Storable as V
import System.Mem (performMajorGC)
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Runtime.Buffer
import HHLO.Runtime.PJRT.Types
import Test.Utils

tests :: TestTree
tests = testGroup "Runtime.Buffer"
    [ testCase "buffer round-trip f32" $ withPJRTCPU $ \api client -> do
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0]
        buf <- toDeviceF32 api client inp [2, 2]
        out <- fromDeviceF32 api buf 4
        out @?= inp
    , testCase "buffer dimensions" $ withPJRTCPU $ \api client -> do
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        buf <- toDeviceF32 api client inp [2, 3]
        dims <- bufferDimensions api buf
        dims @?= [2, 3]
    , testCase "buffer element type f32" $ withPJRTCPU $ \api client -> do
        let inp = V.fromList [1.0, 2.0]
        buf <- toDeviceF32 api client inp [2]
        et <- bufferElementType api buf
        et @?= bufferTypeF32
    , testCase "buffer on-device size" $ withPJRTCPU $ \api client -> do
        let inp = V.fromList [1.0, 2.0, 3.0, 4.0]
        buf <- toDeviceF32 api client inp [2, 2]
        sz <- bufferOnDeviceSize api buf
        sz @?= 16  -- 4 floats * 4 bytes

    -- Regression test for post-session buffer finalizer safety.
    -- Creates a buffer inside a session bracket, returns it so it
    -- outlives the session, then forces GC.  The finalizer must not
    -- segfault even though the client has been destroyed.
    , testCase "buffer finalizer safe after CPU session closes" $ do
        ref <- newIORef False
        _ <- withPJRTCPU $ \api client -> do
            let inp = V.fromList [1.0, 2.0] :: V.Vector Float
            buf <- toDeviceF32 api client inp [2]
            let PJRTBuffer fp = buf
            Conc.addForeignPtrFinalizer fp $ writeIORef ref True
            return buf
        performMajorGC
        -- Poll until the witness finalizer fires (max ~1s).
        let wait n = do
                done <- readIORef ref
                if done
                    then return ()
                    else if n > 100
                        then assertFailure "buffer finalizer did not run"
                        else threadDelay 10000 >> wait (n + 1)
        wait 0
    ]
