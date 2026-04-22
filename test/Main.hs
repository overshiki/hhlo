module Main (main) where

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

main :: IO ()
main = defaultMain $ testGroup "HHLO Tests"
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
    ]
