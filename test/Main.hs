module Main (main) where

import System.Environment (lookupEnv)
import Test.Tasty
import qualified Test.IR.Pretty as Pretty
import qualified Test.IR.Builder as Builder
import qualified Test.EDSL.Ops as EDSLOps
import qualified Test.Runtime.EndToEnd as EndToEnd
import qualified Test.Runtime.EndToEndArithmetic as Arith
import qualified Test.Runtime.EndToEndShape as Shape
import qualified Test.Runtime.EndToEndMatmul as Matmul
import qualified Test.Runtime.EndToEndNN as NN
import qualified Test.Runtime.EndToEndReductions as Reductions
import qualified Test.Runtime.EndToEndDataMovement as DataMovement
import qualified Test.Runtime.Buffer as Buffer
import qualified Test.Runtime.Async as Async
import qualified Test.Runtime.Errors as Errors
import qualified Test.Runtime.EndToEndGPU as EndToEndGPU
import qualified Test.Runtime.BufferGPU as BufferGPU
import qualified Test.Runtime.AsyncGPU as AsyncGPU
import qualified Test.Runtime.MultiGPU as MultiGPU

main :: IO ()
main = do
    mGpu <- lookupEnv "HHLO_TEST_GPU"
    let gpuTests = case mGpu of
            Just "1" ->
                [ EndToEndGPU.tests
                , BufferGPU.tests
                , AsyncGPU.tests
                , MultiGPU.tests
                ]
            _ -> []
    defaultMain $ testGroup "HHLO Tests" $
        [ Pretty.tests
        , Builder.tests
        , EDSLOps.tests
        , EndToEnd.tests
        , Arith.tests
        , Shape.tests
        , Matmul.tests
        , NN.tests
        , Reductions.tests
        , DataMovement.tests
        , Buffer.tests
        , Async.tests
        , Errors.tests
        ] ++ gpuTests
