{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module Test.Autograd.Grad (tests) where

import Prelude hiding (sqrt)
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Pretty
import HHLO.Autograd

import qualified Data.Text as T

tests :: TestTree
tests = testGroup "Autograd.Grad"
    [ testCase "sumOfSquaresGrad" $ do
        let f x = do sq <- multiply x x; sumAll sq
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "should contain stablehlo.multiply" ("stablehlo.multiply" `T.isInfixOf` text)
        assertBool "should contain stablehlo.add" ("stablehlo.add" `T.isInfixOf` text)
    , testCase "sumOfDoublesGrad" $ do
        let f x = do d <- add x x; sumAll d
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "should contain stablehlo.constant" ("stablehlo.constant" `T.isInfixOf` text)
    , testCase "multiplyChainGrad" $ do
        let f x = do
                sq <- multiply x x
                cb <- multiply sq x
                sumAll cb
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "should be non-empty" (not $ T.null text)
    , testCase "mixedOpsGrad" $ do
        let f x = do
                sq <- multiply x x
                s <- sqrt sq
                sumAll s
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "should contain stablehlo.sqrt" ("stablehlo.sqrt" `T.isInfixOf` text)
    ]
