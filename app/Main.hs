{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

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
    putStrLn "=== HHLO End-to-End Demo ==="

    -- 1. Load plugin
    putStrLn "Loading PJRT CPU plugin..."
    api <- withCString "deps/pjrt/libpjrt_cpu.so" $ \path -> do
        alloca $ \apiPtrPtr -> do
            checkError nullPtr $ c_pjrtLoadPlugin path apiPtrPtr
            PJRTApi <$> peek apiPtrPtr
    putStrLn "Plugin loaded."

    -- 2. Create client
    putStrLn "Creating PJRT client..."
    client <- alloca $ \clientPtrPtr -> do
        checkError (unApi api) $ c_pjrtCreateClient (unApi api) clientPtrPtr
        PJRTClient <$> peek clientPtrPtr
    putStrLn "Client created."

    -- 3. Build and compile a simple StableHLO program using the EDSL
    putStrLn "Building program with EDSL..."
    let modu = moduleFromBuilder @'[2,2] @'F32 "main"
            [ FuncArg "arg0" (TensorType [2, 2] F32)
            , FuncArg "arg1" (TensorType [2, 2] F32)
            ]
            $ do
                x <- arg
                y <- arg
                z <- add x y
                return z
    let mlir = render modu
    putStrLn "Generated MLIR:"
    putStrLn (T.unpack mlir)

    putStrLn "Compiling..."
    exec <- compile api client mlir
    putStrLn "Compilation successful."

    -- 4. Create input buffers
    putStrLn "Creating input buffers..."
    let inputX = V.fromList [1.0, 2.0, 3.0, 4.0] :: V.Vector Float
        inputY = V.fromList [5.0, 6.0, 7.0, 8.0] :: V.Vector Float
    bufX <- toDeviceF32 api client inputX [2, 2]
    bufY <- toDeviceF32 api client inputY [2, 2]
    putStrLn "Buffers created."

    -- 5. Execute
    putStrLn "Executing..."
    [bufZ] <- execute api exec [bufX, bufY]
    putStrLn "Execution complete."

    -- 6. Read back result
    result <- fromDeviceF32 api bufZ 4
    putStrLn $ "Result: " ++ show (V.toList result)

    -- 7. Verify
    let expected = [6.0, 8.0, 10.0, 12.0]
    if V.toList result == expected
        then putStrLn "SUCCESS: Results match expected values!"
        else putStrLn $ "FAILURE: Expected " ++ show expected ++ ", got " ++ show (V.toList result)

    -- 8. Cleanup (client only; buffers and executable have ForeignPtr finalizers)
    checkError (unApi api) $ c_pjrtClientDestroy (unApi api) (unClient client)
    putStrLn "Cleanup complete."
  where
    unApi (PJRTApi p) = p
    unClient (PJRTClient p) = p
