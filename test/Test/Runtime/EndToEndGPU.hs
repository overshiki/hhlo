{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndGPU (tests) where

import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.Device

tests :: TestTree
tests = testGroup "EndToEnd.GPU"
    [ testCase "gpu available" gpuAvailableTest
    ]

gpuAvailableTest :: IO ()
gpuAvailableTest = withPJRTGPU $ \api client -> do
    devs <- addressableDevices api client
    case devs of
        [] -> assertFailure "No GPU devices found"
        _  -> do
            mDev <- defaultGPUDevice api client
            case mDev of
                Nothing -> assertFailure "defaultGPUDevice returned Nothing"
                Just _  -> return ()
