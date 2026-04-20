module Main (main) where

import Test.Tasty
import qualified Test.IR.Pretty as Pretty
import qualified Test.Runtime.EndToEnd as EndToEnd

main :: IO ()
main = defaultMain $ testGroup "HHLO Tests"
    [ Pretty.tests
    , EndToEnd.tests
    ]
