{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndGPU (tests) where

import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Runtime.PJRT.Plugin
import HHLO.Runtime.Device
import Test.Runtime.GPUResource (GPUResource(..))

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "EndToEnd.GPU"
    [ testCase "gpu available" $ do
        GPUResource api client _dev <- getGPU
        devs <- addressableDevices api client
        case devs of
            [] -> assertFailure "No GPU devices found"
            _  -> do
                mDev <- defaultGPUDevice api client
                case mDev of
                    Nothing -> assertFailure "defaultGPUDevice returned Nothing"
                    Just _  -> return ()
    ]
