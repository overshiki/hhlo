{-# LANGUAGE OverloadedStrings #-}

module Test.IR.Pretty where

import qualified Data.Text as T
import Test.Tasty
import Test.Tasty.HUnit
import HHLO.IR.AST
import HHLO.IR.Pretty
import HHLO.Core.Types

tests :: TestTree
tests = testGroup "Pretty"
    [ testCase "scalar type" $ do
        render (TensorType [] F32) @?= "tensor<f32>"
    , testCase "2D tensor type" $ do
        render (TensorType [2, 3] F32) @?= "tensor<2x3xf32>"
    , testCase "simple function" $ do
        let fn = Function "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32)
                , FuncArg "arg1" (TensorType [2, 2] F32)
                ]
                [TensorType [2, 2] F32]
                [ValueId 2]
                [ Operation "stablehlo.add" [ValueId 0, ValueId 1]
                    [TensorType [2, 2] F32, TensorType [2, 2] F32] [] [] (ValueId 2) (TensorType [2, 2] F32)
                ]
        let expected =
                "func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
                <> "    %2 = stablehlo.add %0, %1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n"
                <> "    return %2 : tensor<2x2xf32>\n"
                <> "}"
        render fn @?= expected
    ]
