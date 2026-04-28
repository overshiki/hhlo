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
    , testCase "vjpReduceWindow (avgPool)" $ do
        let f x = do
                let windowDims = [1, 2, 2, 1]
                    strides    = [1, 2, 2, 1]
                    padding    = replicate 4 [0, 0]
                initVal <- constant @'[] @'F32 0.0
                y <- reduceWindow windowDims strides padding "stablehlo.add" initVal x
                divisor <- constant @'[] @'F32 4.0
                divisorBC <- broadcastWithDims @'[] @'[1, 1, 1, 1] [] divisor
                z <- divide y divisorBC
                sumAll z
            modu = gradModule @'[1, 4, 4, 1] @'F32 f
            text = render modu
        assertBool "contains reduce_window" ("reduce_window" `T.isInfixOf` text)
        assertBool "contains pad" ("pad" `T.isInfixOf` text)
    , testCase "vjpConvolution" $ do
        let f x = do
                k <- constant @'[2, 2, 1, 1] @'F32 1.0
                y <- conv2d @1 @3 @3 @1 @1 @2 @2 @2 @2 x k
                sumAll y
            modu = gradModule @'[1, 3, 3, 1] @'F32 f
            text = render modu
        assertBool "contains convolution" ("convolution" `T.isInfixOf` text)
        assertBool "contains reverse" ("reverse" `T.isInfixOf` text)
    , testCase "vjpTransposeConvolution" $ do
        let f x = do
                k <- constant @'[2, 2, 1, 1] @'F32 1.0
                y <- transposeConvolution @1 @2 @2 @1 @1 @2 @2 @3 @3 [1, 2, 2, 1] (replicate 2 [0, 0]) x k
                sumAll y
            modu = gradModule @'[1, 2, 2, 1] @'F32 f
            text = render modu
        assertBool "contains convolution" ("convolution" `T.isInfixOf` text)
    ]
