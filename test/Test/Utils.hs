{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module Test.Utils
    ( withPJRTCPU
    , goldenTest
    , e2eTestF32_2arg
    , e2eTestF32_1arg
    , assertThrowsPJRT
    ) where

import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Control.Exception (try)
import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.Compile
import HHLO.Runtime.Execute
import HHLO.Runtime.Buffer

-- | Load PJRT CPU plugin, create client, run action, then destroy client.
withPJRTCPU :: (PJRTApi -> PJRTClient -> IO a) -> IO a
withPJRTCPU action = do
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    result <- action api client

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
    return result

-- | Golden test: compare actual text to expected text.
goldenTest :: String -> T.Text -> T.Text -> TestTree
goldenTest name expected actual =
    testCase name $ actual @?= expected

-- | End-to-end test for F32 ops with two 2x2 inputs.
e2eTestF32_2arg :: String
                -> V.Vector Float   -- ^ input A
                -> V.Vector Float   -- ^ input B
                -> (Tensor '[2, 2] 'F32 -> Tensor '[2, 2] 'F32 -> Builder (Tensor '[2, 2] 'F32))
                -> V.Vector Float   -- ^ expected output
                -> TestTree
e2eTestF32_2arg name inputA inputB fn expected =
    testCase name $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32)
                , FuncArg "arg1" (TensorType [2, 2] F32)
                ]
                $ do
                    x <- arg
                    y <- arg
                    z <- fn x y
                    return z
        exec <- compile api client (render modu)
        bufA <- toDeviceF32 api client inputA [2, 2]
        bufB <- toDeviceF32 api client inputB [2, 2]
        [bufOut] <- execute api exec [bufA, bufB]
        result <- fromDeviceF32 api bufOut 4
        result @?= expected

-- | End-to-end test for F32 ops with one 2x2 input.
e2eTestF32_1arg :: String
                -> V.Vector Float   -- ^ input
                -> (Tensor '[2, 2] 'F32 -> Builder (Tensor '[2, 2] 'F32))
                -> V.Vector Float   -- ^ expected output
                -> TestTree
e2eTestF32_1arg name input fn expected =
    testCase name $ withPJRTCPU $ \api client -> do
        let modu = moduleFromBuilder @'[2, 2] @'F32 "main"
                [ FuncArg "arg0" (TensorType [2, 2] F32)
                ]
                $ do
                    x <- arg
                    z <- fn x
                    return z
        exec <- compile api client (render modu)
        bufIn <- toDeviceF32 api client input [2, 2]
        [bufOut] <- execute api exec [bufIn]
        result <- fromDeviceF32 api bufOut 4
        result @?= expected

-- | Assert that an IO action throws a PJRTException.
assertThrowsPJRT :: String -> IO a -> TestTree
assertThrowsPJRT name action =
    testCase name $ do
        result <- try action
        case result of
            Left (_ :: PJRTException) -> return ()
            Right _ -> assertFailure "Expected PJRTException but action succeeded"

unApi :: PJRTApi -> Ptr PJRTApi
unApi (PJRTApi p) = p

unClient :: PJRTClient -> Ptr PJRTClient
unClient (PJRTClient p) = p
