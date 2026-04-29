{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Runtime.EndToEndArithmeticGPU (tests) where

import Prelude hiding (negate, maximum, minimum, sqrt, sin, cos, tan, floor)
import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.EDSL.Ops
import Test.Utils
import Test.Runtime.GPUResource (GPUResource(..))

inputA :: V.Vector Float
inputA = V.fromList [1.0, 2.0, 3.0, 4.0]

inputB :: V.Vector Float
inputB = V.fromList [5.0, 6.0, 7.0, 8.0]

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "EndToEnd.ArithmeticGPU"
    [ testGroup "Binary element-wise"
        [ e2eTestGPU_F32_2arg "add" inputA inputB add (V.fromList [6.0, 8.0, 10.0, 12.0]) getGPU
        , e2eTestGPU_F32_2arg "sub" inputB inputA sub (V.fromList [4.0, 4.0, 4.0, 4.0]) getGPU
        , e2eTestGPU_F32_2arg "multiply" inputA inputB multiply (V.fromList [5.0, 12.0, 21.0, 32.0]) getGPU
        , e2eTestGPU_F32_2arg "divide" inputB inputA divide (V.fromList [5.0, 3.0, 7.0/3.0, 2.0]) getGPU
        , e2eTestGPU_F32_2arg "maximum" (V.fromList [-1, 2, -3, 4]) (V.fromList [0, 0, 0, 0]) maximum (V.fromList [0, 2, 0, 4]) getGPU
        , e2eTestGPU_F32_2arg "minimum" (V.fromList [-1, 2, -3, 4]) (V.fromList [0, 0, 0, 0]) minimum (V.fromList [-1, 0, -3, 0]) getGPU
        , e2eTestGPU_F32_2arg "pow" (V.fromList [1, 2, 3, 4]) (V.fromList [2, 2, 2, 2]) pow (V.fromList [1, 4, 9, 16]) getGPU
        ]
    , testGroup "Unary element-wise"
        [ e2eTestGPU_F32_1arg "relu positive" inputA relu inputA getGPU
        , e2eTestGPU_F32_1arg "relu negative" (V.fromList [-1, -2, 3, -4]) relu (V.fromList [0, 0, 3, 0]) getGPU
        , e2eTestGPU_F32_1arg "negate" inputA (\x -> negate x) (V.fromList [-1, -2, -3, -4]) getGPU
        , e2eTestGPU_F32_1arg "abs" (V.fromList [-1, -2, 3, -4]) abs' (V.fromList [1, 2, 3, 4]) getGPU
        , e2eTestGPU_F32_1arg "sqrt" (V.fromList [1, 4, 9, 16]) sqrt (V.fromList [1, 2, 3, 4]) getGPU
        , e2eTestGPU_F32_1arg "floor" (V.fromList [1.1, 2.9, 3.0, -1.5]) floor (V.fromList [1, 2, 3, -2]) getGPU
        , e2eTestGPU_F32_1arg "ceil" (V.fromList [1.1, 2.9, 3.0, -1.5]) ceil (V.fromList [2, 3, 3, -1]) getGPU
        ]
    , testGroup "Chain ops"
        [ e2eTestGPU_F32_2arg "(a+b)*(a-b)" inputA inputB
            (\a b -> do s <- add a b; d <- sub a b; multiply s d)
            (V.fromList [-24.0, -32.0, -40.0, -48.0]) getGPU
        ]
    ]
