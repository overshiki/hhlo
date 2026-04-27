{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}

module Test.Autograd.Rules (tests) where

import Prelude hiding (negate)
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Pretty
import HHLO.Autograd

import qualified Data.Text as T

tests :: TestTree
tests = testGroup "Autograd.Rules"
    [ testCase "vjpAdd" $ do
        let f x = do
                y <- add x x
                sumAll y
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "contains func.func" ("func.func" `T.isInfixOf` text)
        assertBool "contains return" ("return" `T.isInfixOf` text)
    , testCase "vjpMultiply" $ do
        let f x = do
                y <- multiply x x
                sumAll y
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "non-empty module" (not $ T.null text)
    , testCase "vjpNegate" $ do
        let f x = do
                y <- negate x
                sumAll y
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "non-empty module" (not $ T.null text)
    , testCase "vjpExponential" $ do
        let f x = do
                y <- exponential x
                sumAll y
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "non-empty module" (not $ T.null text)
    ]
