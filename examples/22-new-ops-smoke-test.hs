{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 22: Smoke test for newly added ops.
--
-- Exercises: transpose, tanh, gelu, concatenate, iota,
--            maxPool, avgPool, globalAvgPool, softmax3D, layerNorm
--
-- Build and run with:
--   LD_LIBRARY_PATH=deps/pjrt:$LD_LIBRARY_PATH cabal run example-new-ops-smoke-test

module Main where

import Data.Int (Int64)
import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

import HHLO.Core.Types
import HHLO.EDSL.Ops hiding (maximum)
import HHLO.IR.AST (FuncArg(..), TensorType(..))
import HHLO.IR.Builder
import HHLO.IR.Pretty
import HHLO.Runtime.PJRT.FFI
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.PJRT.Error
import HHLO.Runtime.Compile
import HHLO.Runtime.Buffer
import HHLO.Runtime.Execute

main :: IO ()
main = do
    putStrLn "=== Example 22: New Ops Smoke Test ==="

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- Test 1: transpose [1,0] on 2x3 matrix
    putStrLn "\n--- Test 1: transpose ---"
    testTranspose api client

    -- Test 2: tanh
    putStrLn "\n--- Test 2: tanh ---"
    testTanh api client

    -- Test 3: gelu
    putStrLn "\n--- Test 3: gelu ---"
    testGelu api client

    -- Test 4: concatenate
    putStrLn "\n--- Test 4: concatenate ---"
    testConcatenate api client

    -- Test 5: iota
    putStrLn "\n--- Test 5: iota ---"
    testIota api client

    -- Test 6: maxPool
    putStrLn "\n--- Test 6: maxPool ---"
    testMaxPool api client

    -- Test 7: avgPool
    putStrLn "\n--- Test 7: avgPool ---"
    testAvgPool api client

    -- Test 8: globalAvgPool
    putStrLn "\n--- Test 8: globalAvgPool ---"
    testGlobalAvgPool api client

    -- Test 9: softmax3D
    putStrLn "\n--- Test 9: softmax3D ---"
    testSoftmax3D api client

    -- Test 10: layerNorm
    putStrLn "\n--- Test 10: layerNorm ---"
    testLayerNorm api client

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
    putStrLn "\n=== All smoke tests completed ==="
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p

testTranspose :: PJRTApi -> PJRTClient -> IO ()
testTranspose api client = do
    let modu = moduleFromBuilder @'[3, 2] @'F32 "main"
            [ FuncArg "x" (TensorType [2, 3] F32) ]
            $ do
                x <- arg @'[2, 3] @'F32
                y <- transpose [1, 0] x
                reshape @'[3, 2] y
    exec <- compile api client (render modu)
    let input = V.fromList [1, 2, 3, 4, 5, 6] :: V.Vector Float
    buf <- toDeviceF32 api client input [2, 3]
    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 6
    let expected = [1, 4, 2, 5, 3, 6] :: [Float]
    checkResult result expected

testTanh :: PJRTApi -> PJRTClient -> IO ()
testTanh api client = do
    let modu = moduleFromBuilder @'[3] @'F32 "main"
            [ FuncArg "x" (TensorType [3] F32) ]
            $ do
                x <- arg @'[3] @'F32
                HHLO.EDSL.Ops.tanh x
    exec <- compile api client (render modu)
    let input = V.fromList [0.0, 0.5, 1.0] :: V.Vector Float
    buf <- toDeviceF32 api client input [3]
    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 3
    let expected = [0.0, 0.46211717, 0.7615942] :: [Float]
    checkApprox result expected 0.001

testGelu :: PJRTApi -> PJRTClient -> IO ()
testGelu api client = do
    let modu = moduleFromBuilder @'[3] @'F32 "main"
            [ FuncArg "x" (TensorType [3] F32) ]
            $ do
                x <- arg @'[3] @'F32
                gelu x
    exec <- compile api client (render modu)
    let input = V.fromList [0.0, 1.0, -1.0] :: V.Vector Float
    buf <- toDeviceF32 api client input [3]
    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 3
    -- GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    let expected = [0.0, 0.8413449, -0.15865526] :: [Float]
    checkApprox result expected 0.01

testConcatenate :: PJRTApi -> PJRTClient -> IO ()
testConcatenate api client = do
    let modu = moduleFromBuilder @'[4] @'F32 "main"
            [ FuncArg "a" (TensorType [2] F32)
            , FuncArg "b" (TensorType [2] F32)
            ]
            $ do
                a <- arg @'[2] @'F32
                b <- arg @'[2] @'F32
                concatenate 0 [a, b]
    exec <- compile api client (render modu)
    let a = V.fromList [1, 2] :: V.Vector Float
        b = V.fromList [3, 4] :: V.Vector Float
    bufA <- toDeviceF32 api client a [2]
    bufB <- toDeviceF32 api client b [2]
    [bufY] <- execute api exec [bufA, bufB]
    result <- fromDeviceF32 api bufY 4
    let expected = [1, 2, 3, 4] :: [Float]
    checkResult result expected

testIota :: PJRTApi -> PJRTClient -> IO ()
testIota api client = do
    let modu = moduleFromBuilder @'[3] @'I64 "main" [] $ do
            iota 0
    exec <- compile api client (render modu)
    [bufY] <- execute api exec []
    result <- fromDevice api bufY 3
    let expected = V.fromList [0, 1, 2] :: V.Vector Int64
    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show (V.toList expected)
    if result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

testMaxPool :: PJRTApi -> PJRTClient -> IO ()
testMaxPool api client = do
    let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
            [ FuncArg "x" (TensorType [1, 4, 4, 1] F32) ]
            $ do
                x <- arg @'[1, 4, 4, 1] @'F32
                maxPool (v2 2 2) (v2 2 2) (p2 (0,0) (0,0)) x
    exec <- compile api client (render modu)
    let input = V.fromList
            [ 1,  2,  3,  4
            , 5,  6,  7,  8
            , 9, 10, 11, 12
            , 13, 14, 15, 16
            ] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 4, 4, 1]
    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 4
    let expected = [6, 8, 14, 16] :: [Float]
    checkResult result expected

testAvgPool :: PJRTApi -> PJRTClient -> IO ()
testAvgPool api client = do
    let modu = moduleFromBuilder @'[1, 2, 2, 1] @'F32 "main"
            [ FuncArg "x" (TensorType [1, 4, 4, 1] F32) ]
            $ do
                x <- arg @'[1, 4, 4, 1] @'F32
                avgPool (v2 2 2) (v2 2 2) x
    exec <- compile api client (render modu)
    let input = V.fromList
            [ 1,  2,  3,  4
            , 5,  6,  7,  8
            , 9, 10, 11, 12
            , 13, 14, 15, 16
            ] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 4, 4, 1]
    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 4
    let expected = [3.5, 5.5, 11.5, 13.5] :: [Float]
    checkResult result expected

testGlobalAvgPool :: PJRTApi -> PJRTClient -> IO ()
testGlobalAvgPool api client = do
    let modu = moduleFromBuilder @'[1, 2] @'F32 "main"
            [ FuncArg "x" (TensorType [1, 2, 2, 2] F32) ]
            $ do
                x <- arg @'[1, 2, 2, 2] @'F32
                globalAvgPool x
    exec <- compile api client (render modu)
    let input = V.fromList
            [ 1, 2
            , 3, 4
            , 5, 6
            , 7, 8
            ] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 2, 2, 2]
    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 2
    let expected = [4.0, 5.0] :: [Float]  -- mean of [1,3,5,7] and [2,4,6,8]
    checkResult result expected

testSoftmax3D :: PJRTApi -> PJRTClient -> IO ()
testSoftmax3D api client = do
    let modu = moduleFromBuilder @'[1, 2, 3] @'F32 "main"
            [ FuncArg "x" (TensorType [1, 2, 3] F32) ]
            $ do
                x <- arg @'[1, 2, 3] @'F32
                softmax3D x
    exec <- compile api client (render modu)
    let input = V.fromList [0, 0, 0, 1, 2, 3] :: V.Vector Float
    buf <- toDeviceF32 api client input [1, 2, 3]
    [bufY] <- execute api exec [buf]
    result <- fromDeviceF32 api bufY 6
    -- Row 0: [0,0,0] -> softmax = [1/3, 1/3, 1/3]
    -- Row 1: [1,2,3] -> exp([1,2,3]) = [2.718, 7.389, 20.085] / 30.193
    let expected = [0.333333, 0.333333, 0.333333
                   , 0.090031, 0.244728, 0.665241] :: [Float]
    checkApprox result expected 0.001

testLayerNorm :: PJRTApi -> PJRTClient -> IO ()
testLayerNorm api client = do
    let modu = moduleFromBuilder @'[1, 2, 3] @'F32 "main"
            [ FuncArg "x" (TensorType [1, 2, 3] F32)
            , FuncArg "g" (TensorType [3] F32)
            , FuncArg "b" (TensorType [3] F32)
            ]
            $ do
                x <- arg @'[1, 2, 3] @'F32
                g <- arg @'[3] @'F32
                b <- arg @'[3] @'F32
                layerNorm x g b
    exec <- compile api client (render modu)
    let input = V.fromList [0, 0, 0, 1, 2, 3] :: V.Vector Float
        gamma = V.fromList [1, 1, 1] :: V.Vector Float
        beta  = V.fromList [0, 0, 0] :: V.Vector Float
    bufX <- toDeviceF32 api client input [1, 2, 3]
    bufG <- toDeviceF32 api client gamma [3]
    bufB <- toDeviceF32 api client beta [3]
    [bufY] <- execute api exec [bufX, bufG, bufB]
    result <- fromDeviceF32 api bufY 6
    -- For row [0,0,0]: mean=0, var=0, normalized=[0,0,0]
    -- For row [1,2,3]: mean=2, var=2/3, std=sqrt(2/3)≈0.816
    --   normalized = [(1-2)/0.816, (2-2)/0.816, (3-2)/0.816] = [-1.225, 0, 1.225]
    let expected = [0.0, 0.0, 0.0, -1.2247, 0.0, 1.2247] :: [Float]
    checkApprox result expected 0.01

checkResult :: V.Vector Float -> [Float] -> IO ()
checkResult result expected = do
    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected
    if V.toList result == expected
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

checkApprox :: V.Vector Float -> [Float] -> Float -> IO ()
checkApprox result expected tol = do
    putStrLn $ "Result:   " ++ show (V.toList result)
    putStrLn $ "Expected: " ++ show expected
    let diffs = zipWith (\a b -> abs (a - b)) (V.toList result) expected
    if all (< tol) diffs
        then putStrLn "✓ PASS"
        else putStrLn $ "✗ FAIL (max diff: " ++ show (maximum diffs) ++ ")"
