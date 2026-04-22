{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Test.EDSL.Ops where

import Prelude hiding (map, maximum, minimum, negate, compare, tanh)
import qualified Data.Text as T
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty

-- | Helper: build a module with one arg, apply a unary op, and check rendered MLIR.
unaryGolden :: String -> (Tensor '[2, 2] 'F32 -> Builder (Tensor '[2, 2] 'F32)) -> String -> TestTree
unaryGolden name op expectedOp =
    testCase name $ do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32) ]
                $ do
                    x <- arg @'[2, 2] @'F32
                    y <- op x
                    return y
        let rendered = render modu
        assertBool ("should contain " ++ expectedOp) $
            T.pack expectedOp `T.isInfixOf` rendered

-- | Helper: build a module with two args, apply a binary op, and check rendered MLIR.
binaryGolden :: String -> (Tensor '[2, 2] 'F32 -> Tensor '[2, 2] 'F32 -> Builder (Tensor '[2, 2] 'F32)) -> String -> TestTree
binaryGolden name op expectedOp =
    testCase name $ do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32)
                , FuncArg "arg1" (TensorType [2, 2] F32)
                ]
                $ do
                    x <- arg @'[2, 2] @'F32
                    y <- arg @'[2, 2] @'F32
                    z <- op x y
                    return z
        let rendered = render modu
        assertBool ("should contain " ++ expectedOp) $
            T.pack expectedOp `T.isInfixOf` rendered

tests :: TestTree
tests = testGroup "EDSL.Ops"
    [ testGroup "Binary element-wise"
        [ binaryGolden "add" add "stablehlo.add"
        , binaryGolden "sub" sub "stablehlo.subtract"
        , binaryGolden "multiply" multiply "stablehlo.multiply"
        , binaryGolden "divide" divide "stablehlo.divide"
        , binaryGolden "maximum" HHLO.EDSL.Ops.maximum "stablehlo.maximum"
        , binaryGolden "minimum" HHLO.EDSL.Ops.minimum "stablehlo.minimum"
        ]
    , testGroup "Unary element-wise"
        [ unaryGolden "relu" relu "stablehlo.maximum"
        , unaryGolden "negate" HHLO.EDSL.Ops.negate "stablehlo.negate"
        , unaryGolden "abs'" abs' "stablehlo.abs"
        , unaryGolden "exponential" exponential "stablehlo.exponential"
        , unaryGolden "logarithm" logarithm "stablehlo.log"
        , unaryGolden "tanh" HHLO.EDSL.Ops.tanh "stablehlo.tanh"
        , unaryGolden "erf" erf "stablehlo.erf"
        ]
    , testGroup "Shape manipulation"
        [ testCase "reshape" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 2] F32) ]
                    $ do
                        x <- arg @'[2, 2] @'F32
                        y <- reshape @'[2, 2] @'[4] x
                        return y
            let rendered = render modu
            assertBool "stablehlo.reshape" $ "stablehlo.reshape" `T.isInfixOf` rendered
        , testCase "transpose" $ do
            let modu = moduleFromBuilder @'[2, 3] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [3, 2] F32) ]
                    $ do
                        x <- arg @'[3, 2] @'F32
                        y <- transpose @'[3, 2] @'[2, 3] [1, 0] x
                        return y
            let rendered = render modu
            assertBool "stablehlo.transpose" $ "stablehlo.transpose" `T.isInfixOf` rendered
        , testCase "broadcast" $ do
            let modu = moduleFromBuilder @'[2, 3] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [3] F32) ]
                    $ do
                        x <- arg @'[3] @'F32
                        y <- broadcastWithDims @'[3] @'[2, 3] [1] x
                        return y
            let rendered = render modu
            assertBool "stablehlo.broadcast_in_dim" $ "stablehlo.broadcast_in_dim" `T.isInfixOf` rendered
        , testCase "concatenate" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2] F32)
                    , FuncArg "arg1" (TensorType [2] F32)
                    ]
                    $ do
                        x <- arg @'[2] @'F32
                        y <- arg @'[2] @'F32
                        z <- concatenate @'[2] @'[4] 0 [x, y]
                        return z
            let rendered = render modu
            assertBool "stablehlo.concatenate" $ "stablehlo.concatenate" `T.isInfixOf` rendered
        , testCase "concatenate2" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2] F32)
                    , FuncArg "arg1" (TensorType [2] F32)
                    ]
                    $ do
                        x <- arg @'[2] @'F32
                        y <- arg @'[2] @'F32
                        z <- concatenate2 @'[2] @'[2] @'[4] 0 x y
                        return z
            let rendered = render modu
            assertBool "stablehlo.concatenate" $ "stablehlo.concatenate" `T.isInfixOf` rendered
        , testCase "iota" $ do
            let modu = moduleFromBuilder @'[4] @'I64 "main"
                    []
                    $ do
                        x <- iota @'[4] 0
                        return x
            let rendered = render modu
            assertBool "stablehlo.iota" $ "stablehlo.iota" `T.isInfixOf` rendered
        ]
    , testGroup "Matmul"
        [ testCase "matmul" $ do
            let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 3] F32)
                    , FuncArg "arg1" (TensorType [3, 2] F32)
                    ]
                    $ do
                        x <- arg @'[2, 3] @'F32
                        y <- arg @'[3, 2] @'F32
                        z <- matmul x y
                        return z
            let rendered = render modu
            assertBool "stablehlo.dot" $ "stablehlo.dot" `T.isInfixOf` rendered
        , testCase "dotGeneral" $ do
            let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 3] F32)
                    , FuncArg "arg1" (TensorType [3, 2] F32)
                    ]
                    $ do
                        x <- arg @'[2, 3] @'F32
                        y <- arg @'[3, 2] @'F32
                        z <- dotGeneral @'[2, 3] @'[3, 2] @'[2, 2] @'F32 [] [] [1] [0] x y
                        return z
            let rendered = render modu
            assertBool "stablehlo.dot_general" $ "stablehlo.dot_general" `T.isInfixOf` rendered
        , testCase "linear" $ do
            let modu = moduleFromBuilder @'[2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [3] F32) ]
                    $ do
                        x <- arg @'[3] @'F32
                        w <- constant @'[3, 2] @'F32 0.5
                        b <- constant @'[2] @'F32 0.1
                        z <- linear x w b
                        return z
            let rendered = render modu
            assertBool "stablehlo.dot" $ "stablehlo.dot" `T.isInfixOf` rendered
            assertBool "stablehlo.add" $ "stablehlo.add" `T.isInfixOf` rendered
        ]
    , testGroup "Reductions"
        [ testCase "reduceSum" $ do
            let modu = moduleFromBuilder @'[] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 3] F32) ]
                    $ do
                        x <- arg @'[2, 3] @'F32
                        y <- reduceSum x
                        return y
            let rendered = render modu
            assertBool "stablehlo.reduce" $ "stablehlo.reduce" `T.isInfixOf` rendered
            assertBool "contains stablehlo.add" $ "stablehlo.add" `T.isInfixOf` rendered
        , testCase "reduceWindow" $ do
            let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 4, 4, 1] F32) ]
                    $ do
                        x <- arg @'[1, 4, 4, 1] @'F32
                        initVal <- constant @'[] @'F32 0.0
                        y <- reduceWindow @'[1, 4, 4, 1] @'[1, 2, 2, 1] [1, 2, 2, 1] [1, 2, 2, 1] [[0, 0], [0, 0], [0, 0], [0, 0]] "stablehlo.add" initVal x
                        return y
            let rendered = render modu
            assertBool "stablehlo.reduce_window" $ "stablehlo.reduce_window" `T.isInfixOf` rendered
        , testCase "maxPool" $ do
            let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 4, 4, 1] F32) ]
                    $ do
                        x <- arg @'[1, 4, 4, 1] @'F32
                        y <- maxPool [2, 2] [2, 2] [[0, 0], [0, 0]] x
                        return y
            let rendered = render modu
            assertBool "reduce_window for maxPool" $ "stablehlo.reduce_window" `T.isInfixOf` rendered
            assertBool "maximum region" $ "stablehlo.maximum" `T.isInfixOf` rendered
        , testCase "avgPool" $ do
            let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 4, 4, 1] F32) ]
                    $ do
                        x <- arg @'[1, 4, 4, 1] @'F32
                        y <- avgPool [2, 2] [2, 2] x
                        return y
            let rendered = render modu
            assertBool "reduce_window for avgPool" $ "stablehlo.reduce_window" `T.isInfixOf` rendered
            assertBool "add region" $ "stablehlo.add" `T.isInfixOf` rendered
        , testCase "globalAvgPool" $ do
            let modu = moduleFromBuilder @'[1, 3] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 4, 4, 3] F32) ]
                    $ do
                        x <- arg @'[1, 4, 4, 3] @'F32
                        y <- globalAvgPool x
                        return y
            let rendered = render modu
            assertBool "contains reduce" $ "stablehlo.reduce" `T.isInfixOf` rendered
        ]
    , testGroup "NN layers"
        [ testCase "conv2d" $ do
            let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 4, 4, 1] F32) ]
                    $ do
                        x <- arg @'[1, 4, 4, 1] @'F32
                        k <- constant @'[3, 3, 1, 1] @'F32 0.5
                        y <- conv2d x k
                        return y
            let rendered = render modu
            assertBool "stablehlo.convolution" $ "stablehlo.convolution" `T.isInfixOf` rendered
        , testCase "conv2dWithPadding" $ do
            let modu = moduleFromBuilder @'[1, 4, 4, 1] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 4, 4, 1] F32) ]
                    $ do
                        x <- arg @'[1, 4, 4, 1] @'F32
                        k <- constant @'[3, 3, 1, 1] @'F32 0.5
                        y <- conv2dWithPadding @1 @4 @4 @1 @1 @3 @3 @4 @4 [1, 1] [[1, 1], [1, 1]] x k
                        return y
            let rendered = render modu
            assertBool "stablehlo.convolution" $ "stablehlo.convolution" `T.isInfixOf` rendered
            assertBool "pad attribute" $ "pad = [[1, 1], [1, 1]]" `T.isInfixOf` rendered
        , testCase "softmax1D" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [4] F32) ]
                    $ do
                        x <- arg @'[4] @'F32
                        y <- softmax1D x
                        return y
            let rendered = render modu
            assertBool "stablehlo.exp" $ "stablehlo.exponential" `T.isInfixOf` rendered
            assertBool "stablehlo.divide" $ "stablehlo.divide" `T.isInfixOf` rendered
        , testCase "softmax2D" $ do
            let modu = moduleFromBuilder @'[2, 4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 4] F32) ]
                    $ do
                        x <- arg @'[2, 4] @'F32
                        y <- softmax2D x
                        return y
            let rendered = render modu
            assertBool "contains reduce" $ "stablehlo.reduce" `T.isInfixOf` rendered
        , testCase "batchNormInference" $ do
            let modu = moduleFromBuilder @'[1, 2, 2, 2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 2, 2, 2] F32) ]
                    $ do
                        x <- arg @'[1, 2, 2, 2] @'F32
                        s <- constant @'[2] @'F32 1.0
                        o <- constant @'[2] @'F32 0.0
                        m <- constant @'[2] @'F32 0.0
                        v <- constant @'[2] @'F32 1.0
                        y <- batchNormInference x s o m v
                        return y
            let rendered = render modu
            assertBool "contains sqrt" $ "stablehlo.sqrt" `T.isInfixOf` rendered
        , testCase "layerNorm" $ do
            let modu = moduleFromBuilder @'[1, 2, 4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 2, 4] F32) ]
                    $ do
                        x <- arg @'[1, 2, 4] @'F32
                        g <- constant @'[4] @'F32 1.0
                        b <- constant @'[4] @'F32 0.0
                        y <- layerNorm x g b
                        return y
            let rendered = render modu
            assertBool "contains reduce" $ "stablehlo.reduce" `T.isInfixOf` rendered
        , testCase "gelu" $ do
            let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 2] F32) ]
                    $ do
                        x <- arg @'[2, 2] @'F32
                        y <- gelu x
                        return y
            let rendered = render modu
            assertBool "contains tanh" $ "stablehlo.tanh" `T.isInfixOf` rendered
        , testCase "transposeConvolution" $ do
            let modu = moduleFromBuilder @'[1, 8, 8, 1] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [1, 4, 4, 1] F32) ]
                    $ do
                        x <- arg @'[1, 4, 4, 1] @'F32
                        k <- constant @'[2, 2, 1, 1] @'F32 0.5
                        y <- transposeConvolution [1, 2, 2, 1] [[1, 1], [1, 1]] x k
                        return y
            let rendered = render modu
            assertBool "stablehlo.convolution" $ "stablehlo.convolution" `T.isInfixOf` rendered
            assertBool "lhs_dilate" $ "lhs_dilate" `T.isInfixOf` rendered
        ]
    , testGroup "Control flow"
        [ testCase "conditional" $ do
            let modu = moduleFromBuilder @'[2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2] F32)
                    , FuncArg "arg1" (TensorType [2] F32)
                    , FuncArg "pred" (TensorType [] Bool)
                    ]
                    $ do
                        t <- arg @'[2] @'F32
                        f <- arg @'[2] @'F32
                        p <- arg @'[] @'Bool
                        y <- conditional p (return t) (return f)
                        return y
            let rendered = render modu
            assertBool "stablehlo.if" $ "stablehlo.if" `T.isInfixOf` rendered
        , testCase "compare" $ do
            let modu = moduleFromBuilder @'[] @'Bool "main"
                    [ FuncArg "arg0" (TensorType [2] F32)
                    , FuncArg "arg1" (TensorType [2] F32)
                    ]
                    $ do
                        x <- arg @'[2] @'F32
                        y <- arg @'[2] @'F32
                        z <- HHLO.EDSL.Ops.compare x y "LT"
                        return z
            let rendered = render modu
            assertBool "stablehlo.compare" $ "stablehlo.compare" `T.isInfixOf` rendered
        ]
    , testGroup "Data movement"
        [ testCase "gather" $ do
            let modu = moduleFromBuilder @'[2, 4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [3, 4] F32) ]
                    $ do
                        x <- arg @'[3, 4] @'F32
                        idx <- constant @'[2] @'I64 0
                        y <- gather x idx [1] [0] [0] 1 [1, 4]
                        return y
            let rendered = render modu
            assertBool "stablehlo.gather" $ "stablehlo.gather" `T.isInfixOf` rendered
        , testCase "scatter" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [4] F32) ]
                    $ do
                        x <- arg @'[4] @'F32
                        idx <- constant @'[1] @'I64 0
                        upd <- constant @'[1] @'F32 9.0
                        y <- scatter x idx upd add [0] [1] [0] 1
                        return y
            let rendered = render modu
            assertBool "stablehlo.scatter" $ "stablehlo.scatter" `T.isInfixOf` rendered
        , testCase "slice" $ do
            let modu = moduleFromBuilder @'[2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [4] F32) ]
                    $ do
                        x <- arg @'[4] @'F32
                        y <- slice x [1] [3] [1]
                        return y
            let rendered = render modu
            assertBool "stablehlo.slice" $ "stablehlo.slice" `T.isInfixOf` rendered
        , testCase "pad" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2] F32) ]
                    $ do
                        x <- arg @'[2] @'F32
                        padVal <- constant @'[] @'F32 0.0
                        y <- pad x padVal [1] [1] [0]
                        return y
            let rendered = render modu
            assertBool "stablehlo.pad" $ "stablehlo.pad" `T.isInfixOf` rendered
        , testCase "dynamicSlice" $ do
            let modu = moduleFromBuilder @'[2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [4] F32) ]
                    $ do
                        x <- arg @'[4] @'F32
                        idx <- constant @'[] @'I64 1
                        y <- dynamicSlice x [idx] [2]
                        return y
            let rendered = render modu
            assertBool "stablehlo.dynamic_slice" $ "stablehlo.dynamic_slice" `T.isInfixOf` rendered
        , testCase "select" $ do
            let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [2, 2] F32)
                    , FuncArg "arg1" (TensorType [2, 2] F32)
                    ]
                    $ do
                        t <- arg @'[2, 2] @'F32
                        f <- arg @'[2, 2] @'F32
                        p <- constant @'[2, 2] @'Bool 1.0
                        y <- select p t f
                        return y
            let rendered = render modu
            assertBool "stablehlo.select" $ "stablehlo.select" `T.isInfixOf` rendered
        , testCase "convert" $ do
            let modu = moduleFromBuilder @'[2] @'I64 "main"
                    [ FuncArg "arg0" (TensorType [2] F32) ]
                    $ do
                        x <- arg @'[2] @'F32
                        y <- convert @'[2] @'F32 @'I64 x
                        return y
            let rendered = render modu
            assertBool "stablehlo.convert" $ "stablehlo.convert" `T.isInfixOf` rendered
        , testCase "sort" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [4] F32) ]
                    $ do
                        x <- arg @'[4] @'F32
                        y <- sort x 0 False (\a b -> HHLO.EDSL.Ops.compare a b "LT")
                        return y
            let rendered = render modu
            assertBool "stablehlo.sort" $ "stablehlo.sort" `T.isInfixOf` rendered
        , testCase "map" $ do
            let modu = moduleFromBuilder @'[4] @'F32 "main"
                    [ FuncArg "arg0" (TensorType [4] F32) ]
                    $ do
                        x <- arg @'[4] @'F32
                        y <- HHLO.EDSL.Ops.map [x] [0] $ \[a] -> multiply a a
                        return y
            let rendered = render modu
            assertBool "stablehlo.map" $ "stablehlo.map" `T.isInfixOf` rendered
        ]
    , testGroup "Constants"
        [ testCase "constant scalar" $ do
            let modu = moduleFromBuilder @'[] @'F32 "main" [] $ do
                    x <- constant @'[] @'F32 3.14
                    return x
            let rendered = render modu
            assertBool "dense scalar" $ "dense<3.14>" `T.isInfixOf` rendered
        , testCase "constant 2D" $ do
            let modu = moduleFromBuilder @'[2, 2] @'F32 "main" [] $ do
                    x <- constant @'[2, 2] @'F32 1.0
                    return x
            let rendered = render modu
            assertBool "dense 2D" $ "dense<[[1.0, 1.0], [1.0, 1.0]]>" `T.isInfixOf` rendered
        , testCase "constant i64" $ do
            let modu = moduleFromBuilder @'[2] @'I64 "main" [] $ do
                    x <- constant @'[2] @'I64 42.0
                    return x
            let rendered = render modu
            assertBool "i64 constant" $ "i64" `T.isInfixOf` rendered
        ]
    ]
