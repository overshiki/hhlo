{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.BufferGPU (tests) where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Device
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

tests :: TestTree
tests = testGroup "Runtime.BufferGPU"
    [ testCase "gpu buffer round-trip f32" gpuRoundTripF32
    , testCase "gpu buffer metadata" gpuBufferMetadata
    ]

gpuRoundTripF32 :: IO ()
gpuRoundTripF32 = withPJRTGPU $ \api client -> do
    mDev <- defaultGPUDevice api client
    dev <- maybe (assertFailure "No GPU found") return mDev

    let input = V.fromList [1, 2, 3, 4, 5, 6] :: V.Vector Float
    buf <- toDeviceOn api client dev input [2, 3] bufferTypeF32
    result <- fromDeviceF32 api buf 6
    result @?= input

gpuBufferMetadata :: IO ()
gpuBufferMetadata = withPJRTGPU $ \api client -> do
    mDev <- defaultGPUDevice api client
    dev <- maybe (assertFailure "No GPU found") return mDev

    let input = V.fromList [1..12] :: V.Vector Float
    buf <- toDeviceOn api client dev input [3, 4] bufferTypeF32

    dims <- bufferDimensions api buf
    dims @?= [3, 4]

    et <- bufferElementType api buf
    et @?= bufferTypeF32

    sz <- bufferOnDeviceSize api buf
    sz @?= (12 * 4)  -- 12 floats * 4 bytes
