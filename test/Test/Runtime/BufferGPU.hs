{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.BufferGPU (tests) where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Buffer
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
    ]
