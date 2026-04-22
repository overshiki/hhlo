{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.Buffer where

import qualified Data.Vector.Storable as V
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
    ]
