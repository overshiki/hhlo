module Main (main) where

import System.Environment (lookupEnv)
import Test.Tasty

import qualified Test.IR.Pretty as Pretty
import qualified Test.IR.Builder as Builder
import qualified Test.EDSL.Ops as EDSLOps
import qualified Test.Autograd.Grad as AutogradGrad
import qualified Test.Autograd.Rules as AutogradRules
import qualified Test.Runtime.EndToEnd as EndToEnd
import qualified Test.Runtime.EndToEndArithmetic as Arith
import qualified Test.Runtime.EndToEndShape as Shape
import qualified Test.Runtime.EndToEndMatmul as Matmul
import qualified Test.Runtime.EndToEndNN as NN
import qualified Test.Runtime.EndToEndReductions as Reductions
import qualified Test.Runtime.EndToEndDataMovement as DataMovement
import qualified Test.Runtime.EndToEndMultiValue as MultiValue
import qualified Test.Runtime.EndToEndSession as Session
import qualified Test.Runtime.EndToEndAutograd as Autograd
import qualified Test.Runtime.EndToEndDynamic as Dynamic
import qualified Test.Runtime.EndToEndDynamicGPU as DynamicGPU
import qualified Test.Runtime.Buffer as Buffer
import qualified Test.Runtime.Async as Async
import qualified Test.Runtime.Errors as Errors

import Test.Runtime.GPUResource (acquireGPU, releaseGPU)
import qualified Test.Runtime.EndToEndGPU as EndToEndGPU
import qualified Test.Runtime.BufferGPU as BufferGPU
import qualified Test.Runtime.AsyncGPU as AsyncGPU
import qualified Test.Runtime.MultiGPU as MultiGPU
import qualified Test.Runtime.EndToEndArithmeticGPU as ArithGPU
import qualified Test.Runtime.EndToEndShapeGPU as ShapeGPU
import qualified Test.Runtime.EndToEndMatmulGPU as MatmulGPU
import qualified Test.Runtime.EndToEndNNGPU as NNGPU
import qualified Test.Runtime.EndToEndReductionsGPU as ReductionsGPU
import qualified Test.Runtime.EndToEndDataMovementGPU as DataMovementGPU
import qualified Test.Runtime.EndToEndMultiValueGPU as MultiValueGPU
import qualified Test.Runtime.EndToEndAutogradGPU as AutogradGPU
import qualified Test.Runtime.EndToEndSessionGPU as SessionGPU

cpuTests :: [TestTree]
cpuTests =
    [ Pretty.tests
    , Builder.tests
    , EDSLOps.tests
    , AutogradGrad.tests
    , AutogradRules.tests
    , EndToEnd.tests
    , Arith.tests
    , Shape.tests
    , Matmul.tests
    , NN.tests
    , Reductions.tests
    , DataMovement.tests
    , MultiValue.tests
    , Session.tests
    , Autograd.tests
    , Dynamic.tests
    , Buffer.tests
    , Async.tests
    , Errors.tests
    ]

main :: IO ()
main = do
    mGpu <- lookupEnv "HHLO_TEST_GPU"
    case mGpu of
        Just "1" ->
            defaultMain $ withResource acquireGPU releaseGPU $ \getGPU ->
                testGroup "HHLO Tests" $ cpuTests ++
                    [ testGroup "GPU"
                        [ EndToEndGPU.tests getGPU
                        , BufferGPU.tests getGPU
                        , AsyncGPU.tests getGPU
                        , MultiGPU.tests getGPU
                        , ArithGPU.tests getGPU
                        , ShapeGPU.tests getGPU
                        , MatmulGPU.tests getGPU
                        , NNGPU.tests getGPU
                        , ReductionsGPU.tests getGPU
                        , DataMovementGPU.tests getGPU
                        , MultiValueGPU.tests getGPU
                        , AutogradGPU.tests getGPU
                        , SessionGPU.tests getGPU
                        , DynamicGPU.tests getGPU
                        ]
                    ]
        _ -> defaultMain $ testGroup "HHLO Tests" cpuTests
