{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
module Test.Autograd.Rules (tests) where

import Prelude hiding (negate)
import Test.Tasty
import Test.Tasty.HUnit

import Data.Proxy (Proxy(..))
import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), Module)
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Autograd
import HHLO.ShapeCheck

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
                let windowDims = v4 1 2 2 1
                    strides    = v4 1 2 2 1
                    padding    = v4 (0,0) (0,0) (0,0) (0,0)
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
                y <- transposeConvolution @1 @2 @2 @1 @1 @2 @2 @2 @2 (v2 2 2) (p2 (0,0) (0,0)) x k
                sumAll y
            modu = gradModule @'[1, 2, 2, 1] @'F32 f
            text = render modu
        assertBool "contains convolution" ("convolution" `T.isInfixOf` text)
    , testCase "vjpConvolution checkModule" $ do
        let f x = do
                k <- constant @'[2, 2, 1, 1] @'F32 1.0
                y <- conv2d @1 @3 @3 @1 @1 @2 @2 @2 @2 x k
                sumAll y
            modu = gradModule @'[1, 3, 3, 1] @'F32 f
        case checkModule modu of
            Left err -> assertFailure $ show err
            Right () -> return ()
    , testCase "vjpTransposeConvolution checkModule" $ do
        let f x = do
                k <- constant @'[2, 2, 1, 1] @'F32 1.0
                y <- transposeConvolution @1 @2 @2 @1 @1 @2 @2 @2 @2 (v2 2 2) (p2 (0,0) (0,0)) x k
                sumAll y
            modu = gradModule @'[1, 2, 2, 1] @'F32 f
        case checkModule modu of
            Left err -> assertFailure $ show err
            Right () -> return ()
    , testCase "vjpConvolution through moduleFromBuilder" $ do
        let modu :: Module
            modu = moduleFromBuilder @'[1, 3, 3, 1] @'F32 "main"
                [FuncArg "x" (tensorType (Proxy @'[1, 3, 3, 1]) (Proxy @'F32))]
                $ do
                    x <- arg @'[1, 3, 3, 1] @'F32
                    k <- constant @'[2, 2, 1, 1] @'F32 1.0
                    let f z = do
                            y <- conv2d @1 @3 @3 @1 @1 @2 @2 @2 @2 z k
                            sumAll y
                    seed <- constant @'[] @'F32 1.0
                    gradX <- vjp f x seed
                    return gradX
            text = render modu
        assertBool "contains convolution" ("convolution" `T.isInfixOf` text)
    , testCase "vjpTransposeConvolution through moduleFromBuilder" $ do
        let modu :: Module
            modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                [FuncArg "x" (tensorType (Proxy @'[1, 2, 2, 1]) (Proxy @'F32))]
                $ do
                    x <- arg @'[1, 2, 2, 1] @'F32
                    k <- constant @'[2, 2, 1, 1] @'F32 1.0
                    let f z = do
                            y <- transposeConvolution @1 @2 @2 @1 @1 @2 @2 @2 @2 (v2 2 2) (p2 (0,0) (0,0)) z k
                            sumAll y
                    seed <- constant @'[] @'F32 1.0
                    gradX <- vjp f x seed
                    return gradX
            text = render modu
        assertBool "contains convolution" ("convolution" `T.isInfixOf` text)
    , testCase "gradModule2" $ do
        let f x y = do
                z <- multiply x y
                sumAll z
            modu = gradModule2 @'[2] @'F32 @'[2] @'F32 f
            text = render modu
        assertBool "contains func.func" ("func.func" `T.isInfixOf` text)
        assertBool "two results" $ (length $ filter (=="->") $ T.chunksOf 2 text) >= 1
    , testCase "vjpCustomCall" $ do
        let f x = do
                y <- customVJP1 "myinc" "myinc_bwd" [x] "" False
                sumAll y
            modu = gradModule @'[2] @'F32 f
            text = render modu
        assertBool "contains forward myinc" ("call_target_name = \"myinc\"" `T.isInfixOf` text)
        assertBool "contains hhlo.vjp_target" ("hhlo.vjp_target = \"myinc_bwd\"" `T.isInfixOf` text)
        assertBool "contains backward myinc_bwd" ("call_target_name = \"myinc_bwd\"" `T.isInfixOf` text)
    ]
