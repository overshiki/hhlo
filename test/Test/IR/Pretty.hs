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
    [ testGroup "Types"
        [ testCase "scalar type" $ do
            render (TensorType [] F32) @?= "tensor<f32>"
        , testCase "2D tensor type" $ do
            render (TensorType [2, 3] F32) @?= "tensor<2x3xf32>"
        , testCase "4D tensor type" $ do
            render (TensorType [1, 8, 8, 16] F32) @?= "tensor<1x8x8x16xf32>"
        , testCase "i64 tensor type" $ do
            render (TensorType [2, 3] I64) @?= "tensor<2x3xi64>"
        , testCase "bool tensor type" $ do
            render (TensorType [2, 3] Bool) @?= "tensor<2x3xi1>"
        ]
    , testGroup "Custom op formats"
        [ testCase "simple add" $ do
            let fn = Function "main"
                    [ FuncArg "arg0" (TensorType [2, 2] F32)
                    , FuncArg "arg1" (TensorType [2, 2] F32)
                    ]
                    [TensorType [2, 2] F32]
                    [ValueId 2]
                    [ Operation "stablehlo.add" [ValueId 0, ValueId 1]
                        [TensorType [2, 2] F32, TensorType [2, 2] F32] [] [] [ValueId 2] [TensorType [2, 2] F32]
                    ]
            let expected =
                    "func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
                    <> "    %2 = stablehlo.add %0, %1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n"
                    <> "    return %2 : tensor<2x2xf32>\n"
                    <> "}"
            render fn @?= expected
        , testCase "broadcast_in_dim trailing format" $ do
            let op = Operation "stablehlo.broadcast_in_dim" [ValueId 0]
                    [TensorType [3] F32]
                    [AttrIntList "broadcast_dimensions" [1]]
                    [] [ValueId 1] [TensorType [2, 3] F32]
            let rendered = render op
            assertBool "should contain trailing dims" $
                ", dims = [1]" `T.isInfixOf` rendered
        , testCase "compare inline direction" $ do
            let op = Operation "stablehlo.compare" [ValueId 0, ValueId 1]
                    [TensorType [2] F32, TensorType [2] F32]
                    [AttrString "comparison_direction" "LT"]
                    [] [ValueId 2] [TensorType [2] Bool]
            let rendered = render op
            assertBool "should contain inline direction" $
                "\"LT\"" `T.isInfixOf` rendered
        , testCase "return generic form" $ do
            let op = Operation "stablehlo.return" [ValueId 0]
                    [TensorType [] F32] [] [] [ValueId 0] [TensorType [] F32]
            let rendered = render op
            assertBool "should be generic form" $
                "\"stablehlo.return\"" `T.isInfixOf` rendered
        , testCase "custom_call with @symbol" $ do
            let op = Operation "stablehlo.custom_call" [ValueId 0, ValueId 1]
                    [TensorType [4] F32, TensorType [4] F32]
                    [ AttrString "call_target_name" "vector_add"
                    , AttrBool   "has_side_effect"  False
                    , AttrString "backend_config"   ""
                    , AttrInt    "api_version"      1
                    ] [] [ValueId 2] [TensorType [4] F32]
            let rendered = render op
            assertBool "should contain @symbol prefix" $
                "stablehlo.custom_call @vector_add(" `T.isInfixOf` rendered
            assertBool "should contain call_target_name attr" $
                "call_target_name = \"vector_add\"" `T.isInfixOf` rendered
            assertBool "should contain has_side_effect attr" $
                "has_side_effect = false" `T.isInfixOf` rendered
            assertBool "should contain api_version attr" $
                "api_version = 1 : i64" `T.isInfixOf` rendered
            assertBool "should end with function type" $
                ": (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>" `T.isSuffixOf` rendered
        , testCase "custom_call without operands" $ do
            let op = Operation "stablehlo.custom_call" []
                    []
                    [ AttrString "call_target_name" "rng_seed"
                    , AttrBool   "has_side_effect"  False
                    ] [] [ValueId 0] [TensorType [] I64]
            let rendered = render op
            assertBool "should contain @symbol with empty parens" $
                "stablehlo.custom_call @rng_seed()" `T.isInfixOf` rendered
        ]
    , testGroup "Constants"
        [ testCase "scalar constant" $ do
            let op = Operation "stablehlo.constant" []
                    [] [AttrDenseElements [] F32 [3.0]] [] [ValueId 0] [TensorType [] F32]
            let rendered = render op
            assertBool "dense scalar" $ "dense<3.0>" `T.isInfixOf` rendered
        , testCase "1D constant" $ do
            let op = Operation "stablehlo.constant" []
                    [] [AttrDenseElements [3] F32 [1.0, 2.0, 3.0]] [] [ValueId 0] [TensorType [3] F32]
            let rendered = render op
            assertBool "dense 1D" $ "dense<[1.0, 2.0, 3.0]>" `T.isInfixOf` rendered
        , testCase "2D constant" $ do
            let op = Operation "stablehlo.constant" []
                    [] [AttrDenseElements [2, 2] F32 [1.0, 2.0, 3.0, 4.0]] [] [ValueId 0] [TensorType [2, 2] F32]
            let rendered = render op
            assertBool "dense 2D" $ "dense<[[1.0, 2.0], [3.0, 4.0]]>" `T.isInfixOf` rendered
        ]
    , testGroup "Multi-result ops"
        [ testCase "two results" $ do
            let op = Operation "stablehlo.rng_bit_generator" [ValueId 0]
                    [TensorType [2] UI64]
                    [AttrRaw "rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>"]
                    [] [ValueId 1, ValueId 2] [TensorType [2] UI64, TensorType [4] UI64]
            let rendered = render op
            assertBool "two result vids" $ "%1, %2 =" `T.isInfixOf` rendered
            assertBool "rng_bit_generator" $ "stablehlo.rng_bit_generator" `T.isInfixOf` rendered
        ]
    ]
