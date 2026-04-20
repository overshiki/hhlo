{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Example 3: Chained operations.
--
-- Computes: result = (a + b) * (a - b)
-- Which is equivalent to: a² - b²  (difference of squares)

module Main where

import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Foreign.C
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

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

main :: IO ()
main = do
    putStrLn "=== Example 3: Chained Operations ==="
    putStrLn "Compute: (a + b) * (a - b) = a² - b²"

    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr

    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr

    -- Build program
    let modu = moduleFromBuilder @'[2,2] @'F32 "main"
            [ FuncArg "a" (TensorType [2, 2] F32)
            , FuncArg "b" (TensorType [2, 2] F32)
            ]
            $ do
                a <- arg
                b <- arg
                s <- add a b
                d <- sub a b
                r <- multiply s d
                return r

    putStrLn "Generated MLIR:"
    putStrLn (T.unpack $ render modu)

    exec <- compile api client (render modu)

    let inputA = V.fromList [5, 4, 3, 2] :: V.Vector Float
        inputB = V.fromList [1, 2, 1, 2] :: V.Vector Float
    bufA <- toDeviceF32 api client inputA [2, 2]
    bufB <- toDeviceF32 api client inputB [2, 2]

    [bufR] <- execute api exec [bufA, bufB]
    result <- fromDeviceF32 api bufR 4

    putStrLn $ "a:        " ++ show (V.toList inputA)
    putStrLn $ "b:        " ++ show (V.toList inputB)
    putStrLn $ "Result:   " ++ show (V.toList result)
    -- (5+1)*(5-1)=24, (4+2)*(4-2)=12, (3+1)*(3-1)=8, (2+2)*(2-2)=0
    putStrLn $ "Expected: [24.0,12.0,8.0,0.0]"

    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
