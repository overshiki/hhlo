{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.ModuleBuilder (tests) where

import Prelude hiding (compare, map)
import qualified Data.Text as T
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (Module)
import HHLO.IR.Builder (Tensor)
import HHLO.IR.Pretty (render)
import HHLO.ModuleBuilder

-- | A simple 1-input, 1-output module.
addModule :: Module
addModule = buildModule @1 @1 "add_one" $ \x -> do
    one <- constant @'[2] @'F32 1.0
    add x one

-- | A 2-input, 1-output module.
mulModule :: Module
mulModule = buildModule @2 @1 "mul" $ \(x :: Tensor '[2] F32) (y :: Tensor '[2] F32) -> do
    multiply x y

-- | A 1-input, 2-output module.
splitModule :: Module
splitModule = buildModule @1 @2 "split" $ \(x :: Tensor '[2] F32) -> do
    y <- add x x
    z <- multiply x x
    returnTuple2 y z

-- | A 2-input, 2-output module.
swapModule :: Module
swapModule = buildModule @2 @2 "swap" $ \(x :: Tensor '[2] F32) (y :: Tensor '[3] F32) -> do
    returnTuple2 y x

-- | A 3-input, 1-output module.
tripleAddModule :: Module
tripleAddModule = buildModule @3 @1 "triple_add" $ \(x :: Tensor '[2] F32) (y :: Tensor '[2] F32) (z :: Tensor '[2] F32) -> do
    xy <- add x y
    add xy z

tests :: TestTree
tests = testGroup "ModuleBuilder"
    [ testCase "buildModule @1 @1 renders stablehlo.add" $ do
        let text = render addModule
        assertBool "should contain func.func" $ "func.func" `T.isInfixOf` text
        assertBool "should contain stablehlo.add" $ "stablehlo.add" `T.isInfixOf` text
    , testCase "buildModule @2 @1 renders stablehlo.multiply" $ do
        let text = render mulModule
        assertBool "should contain func.func" $ "func.func" `T.isInfixOf` text
        assertBool "should contain stablehlo.multiply" $ "stablehlo.multiply" `T.isInfixOf` text
    , testCase "buildModule @1 @2 renders two results" $ do
        let text = render splitModule
        assertBool "should contain func.func" $ "func.func" `T.isInfixOf` text
        assertBool "should contain stablehlo.add" $ "stablehlo.add" `T.isInfixOf` text
        assertBool "should contain stablehlo.multiply" $ "stablehlo.multiply" `T.isInfixOf` text
    , testCase "buildModule @2 @2 renders two results" $ do
        let text = render swapModule
        assertBool "should contain func.func" $ "func.func" `T.isInfixOf` text
    , testCase "buildModule @3 @1 renders stablehlo.add twice" $ do
        let text = render tripleAddModule
        assertBool "should contain func.func" $ "func.func" `T.isInfixOf` text
        assertBool "should contain stablehlo.add" $ "stablehlo.add" `T.isInfixOf` text
    ]
