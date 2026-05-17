{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.BufferGPU (tests) where

import Control.Concurrent (threadDelay)
import Data.IORef (newIORef, readIORef, writeIORef)
import qualified Foreign.Concurrent as Conc (addForeignPtrFinalizer)
import qualified Data.Vector.Storable as V
import System.Mem (performMajorGC)
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Buffer
import HHLO.Runtime.Device (addressableDevices)
import HHLO.Runtime.PJRT.Plugin (withPJRTGPU)
import Test.Runtime.GPUResource (GPUResource(..))

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "Runtime.BufferGPU"
    [ testCase "gpu buffer round-trip f32" $ do
        GPUResource api client dev <- getGPU
        let input = V.fromList [1, 2, 3, 4, 5, 6] :: V.Vector Float
        buf <- toDeviceOn api client dev input [2, 3] bufferTypeF32
        result <- fromDeviceF32 api buf 6
        result @?= input

    , testCase "gpu buffer metadata" $ do
        GPUResource api client dev <- getGPU
        let input = V.fromList [1..12] :: V.Vector Float
        buf <- toDeviceOn api client dev input [3, 4] bufferTypeF32
        dims <- bufferDimensions api buf
        dims @?= [3, 4]
        et <- bufferElementType api buf
        et @?= bufferTypeF32
        sz <- bufferOnDeviceSize api buf
        sz @?= (12 * 4)

    -- Regression test for the post-session buffer finalizer segfault.
    -- Without the registry fix, this crashes because the GPU buffer's
    -- internal CUDA references become dangling after PJRT_Client_Destroy.
    , testCase "buffer finalizer safe after GPU session closes" $ do
        ref <- newIORef False
        _ <- withPJRTGPU $ \api client -> do
            devs <- addressableDevices api client
            let dev = head devs
            let inp = V.fromList [1.0, 2.0] :: V.Vector Float
            buf <- toDeviceOn api client dev inp [2] bufferTypeF32
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
