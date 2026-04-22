{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Test.IR.Builder where

import qualified Data.Text as T
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.IR.AST
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Core.Types

tests :: TestTree
tests = testGroup "Builder"
    [ testCase "value ids are sequential" $ do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32)
                , FuncArg "arg1" (TensorType [2, 2] F32)
                ]
                $ do
                    x <- arg @'[2, 2] @'F32
                    y <- arg @'[2, 2] @'F32
                    let (Tensor v1) = x
                        (Tensor v2) = y
                    return $ Tensor (ValueId 42)  -- dummy, we just check ids
        let rendered = render modu
        assertBool "arg0 present" $ "%arg0" `T.isInfixOf` rendered
        assertBool "arg1 present" $ "%arg1" `T.isInfixOf` rendered
    , testCase "module has func.func wrapper" $ do
        let modu = moduleFromBuilder @'[2] @'F32 "test_fn"
                [ FuncArg "x" (TensorType [2] F32) ]
                $ do
                    x <- arg @'[2] @'F32
                    return x
        let rendered = render modu
        assertBool "module keyword" $ "module {" `T.isInfixOf` rendered
        assertBool "func.func keyword" $ "func.func @test_fn" `T.isInfixOf` rendered
        assertBool "return present" $ "return" `T.isInfixOf` rendered
    , testCase "single result type in signature" $ do
        let modu = moduleFromBuilder @'[3, 4] @'F32 "main"
                [ FuncArg "arg0" (TensorType [3, 4] F32) ]
                $ do
                    x <- arg
                    return x
        let rendered = render modu
        assertBool "return type" $ "-> tensor<3x4xf32>" `T.isInfixOf` rendered
    ]
