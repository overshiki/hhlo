{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Test.Runtime.EndToEnd where

import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.Plugin (withPJRTCPU)
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

tests :: TestTree
tests = testGroup "EndToEnd"
    [ testCase "add two tensors on CPU" $ withPJRTCPU $ \api client -> do
        -- 1. Build and compile a program using the EDSL
        let modu = moduleFromBuilder @'[2,2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [Just 2, Just 2] F32)
                , FuncArg "arg1" (TensorType [Just 2, Just 2] F32)
                ]
                $ do
                    x <- arg
                    y <- arg
                    z <- add x y
                    return z
        let mlir = render modu
        exec <- compile api client mlir

        -- 2. Create input buffers
        let inputX = V.fromList [1.0, 2.0, 3.0, 4.0] :: V.Vector Float
            inputY = V.fromList [5.0, 6.0, 7.0, 8.0] :: V.Vector Float
        bufX <- toDeviceF32 api client inputX [2, 2]
        bufY <- toDeviceF32 api client inputY [2, 2]

        -- 3. Execute
        [bufZ] <- execute api exec [bufX, bufY]

        -- 4. Read back result
        result <- fromDeviceF32 api bufZ 4

        -- 5. Verify
        result @?= V.fromList [6.0, 8.0, 10.0, 12.0]
    ]


