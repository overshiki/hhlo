{-# LANGUAGE OverloadedStrings #-}

module Test.Runtime.EndToEnd where

import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Foreign.C
import Foreign.Ptr
import Foreign.Marshal.Alloc (alloca)
import Foreign.Storable (peek)
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

tests :: TestTree
tests = testGroup "EndToEnd"
    [ testCase "add two tensors on CPU" $ do
        -- 1. Load plugin
        api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
            alloca $ \apiPtrPtr -> do
                checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
                PJRTApi <$> peek apiPtrPtr

        -- 2. Create client
        client <- alloca $ \clientPtrPtr -> do
            checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
            PJRTClient <$> peek clientPtrPtr

        -- 3. Compile a simple StableHLO program
        let mlir = T.unlines
                [ "func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {"
                , "    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>"
                , "    return %0 : tensor<2x2xf32>"
                , "}"
                ]
        exec <- compile api client mlir

        -- 4. Create input buffers
        let inputX = V.fromList [1.0, 2.0, 3.0, 4.0] :: V.Vector Float
            inputY = V.fromList [5.0, 6.0, 7.0, 8.0] :: V.Vector Float
        bufX <- toDeviceF32 api client inputX [2, 2]
        bufY <- toDeviceF32 api client inputY [2, 2]

        -- 5. Execute
        [bufZ] <- execute api exec [bufX, bufY]

        -- 6. Read back result
        result <- fromDeviceF32 api bufZ 4

        -- 7. Verify
        result @?= V.fromList [6.0, 8.0, 10.0, 12.0]

        -- 8. Cleanup
        destroyBuffer api bufX
        destroyBuffer api bufY
        destroyBuffer api bufZ
        checkError (unApi api) $ c_pjrtLoadedExecutableDestroy (unApi api) (unExec exec)
        checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
    ]

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p

unExec :: PJRTExecutable -> Ptr PJRTExecutable
unExec (PJRTExecutable p) = p
