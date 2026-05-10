{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Test.Runtime.EndToEndSessionGPU (tests) where

import Prelude hiding (compare)
import qualified Data.Vector.Storable as V
import Test.Tasty
import Test.Tasty.HUnit

import HHLO.Core.Types
import HHLO.EDSL.Ops
import HHLO.IR.Builder (Tensor, moduleFromBuilder)
import HHLO.ModuleBuilder
import HHLO.Session
import HHLO.IR.AST (Module)
import Test.Runtime.GPUResource (GPUResource(..))

addOneModule :: Module
addOneModule = buildModule @1 @1 "add_one" $ \x -> do
    one <- constant @'[2] @'F32 1.0
    add x one

constantModule :: Module
constantModule = moduleFromBuilder @'[2] @'F32 "main" [] $ do
    constant @'[2] @'F32 42.0

mulModule :: Module
mulModule = buildModule @2 @1 "mul" $ \(x :: Tensor '[2] F32) (y :: Tensor '[2] F32) -> do
    multiply x y

splitModule :: Module
splitModule = buildModule @1 @2 "split" $ \(x :: Tensor '[2] F32) -> do
    y <- add x x
    z <- multiply x x
    returnTuple2 y z

tests :: IO GPUResource -> TestTree
tests getGPU = testGroup "EndToEnd.SessionGPU"
    [ testCase "run zero-input module on GPU" $ do
        GPUResource api client dev <- getGPU
        let sess = sessionFrom api client dev
        compiled <- compile sess constantModule
        (result :: HostTensor '[2] 'F32) <- run sess compiled ()
        let vec = hostToVector result
        vec @?= V.fromList [42.0, 42.0]

    , testCase "run single-input module on GPU" $ do
        GPUResource api client dev <- getGPU
        let sess = sessionFrom api client dev
        compiled <- compile sess addOneModule
        (result :: HostTensor '[2] 'F32) <- run sess compiled (hostFromList @'[2] @'F32 [1.0, 2.0])
        let vec = hostToVector result
        vec @?= V.fromList [2.0, 3.0]

    , testCase "run two-input module on GPU" $ do
        GPUResource api client dev <- getGPU
        let sess = sessionFrom api client dev
        compiled <- compile sess mulModule
        (result :: HostTensor '[2] 'F32) <- run sess compiled
            ( hostFromList @'[2] @'F32 [2.0, 3.0]
            , hostFromList @'[2] @'F32 [4.0, 5.0]
            )
        let vec = hostToVector result
        vec @?= V.fromList [8.0, 15.0]

    , testCase "run two-output module on GPU" $ do
        GPUResource api client dev <- getGPU
        let sess = sessionFrom api client dev
        compiled <- compile sess splitModule
        ((r1 :: HostTensor '[2] 'F32), (r2 :: HostTensor '[2] 'F32)) <-
            run sess compiled (hostFromList @'[2] @'F32 [2.0, 3.0])
        hostToVector r1 @?= V.fromList [4.0, 6.0]
        hostToVector r2 @?= V.fromList [4.0, 9.0]

    , testCase "runAsync is equivalent to run on GPU" $ do
        GPUResource api client dev <- getGPU
        let sess = sessionFrom api client dev
        compiled <- compile sess addOneModule
        (result :: HostTensor '[2] 'F32) <- runAsync sess compiled (hostFromList @'[2] @'F32 [5.0, 6.0])
        awaitOutputs sess result
        let vec = hostToVector result
        vec @?= V.fromList [6.0, 7.0]
    ]
