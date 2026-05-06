{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}

-- | High-level session API for compiling and executing StableHLO modules.
--
-- This module eliminates the boilerplate of PJRT plugin discovery,
-- buffer management, and shape bookkeeping. A typical program looks like:
--
-- > main = withCPU $ \sess -> do
-- >     compiled <- compile sess myModule
-- >     result   <- run sess compiled (hostFromList @'[2] [1.0, 2.0])
-- >     print (hostToList result)
module HHLO.Session
    ( -- * Session lifecycle
      Session(..)
    , withCPU
    , withGPU
    , withGPUDevice
    , sessionFrom
      -- * Compilation
    , Compiled
    , compile
      -- * Execution
    , run
    , runAsync
    , awaitOutputs
      -- * Host-side typed tensors
    , HostTensor
    , hostFromList
    , hostFromVector
    , hostFromListSafe
    , hostFromVectorSafe
    , hostToList
    , hostToVector
      -- * Dynamic-shape host tensors
    , DynamicHostTensor(..)
    , dynamicHostFromVector
    , dynamicHostToVector
    , runDynamic
    , runDynamicAsync
    ) where

import Control.Monad (when)
import Data.Either (fromRight)
import Data.Int (Int64)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Vector.Storable as V
import Foreign.C (CInt)
import GHC.TypeLits
import System.IO.Unsafe (unsafePerformIO)

import HHLO.IR.AST (ttDType)

import HHLO.Core.Types
import HHLO.IR.AST (Module)
import HHLO.IR.Builder (KnownDType(..))
import HHLO.IR.Pretty (render)
import HHLO.Runtime.PJRT.Plugin (withPJRT, getPluginPath)
import HHLO.Runtime.PJRT.Types
import HHLO.Runtime.Compile (CompileOptions(..), defaultCompileOptions, compileWithOptions)
import HHLO.Runtime.Execute (execute)
import qualified HHLO.Runtime.Async as Async
import qualified HHLO.Runtime.Buffer as Buf
import qualified HHLO.Runtime.Device as Dev

-- ---------------------------------------------------------------------------
-- Session
-- ---------------------------------------------------------------------------

-- | A runtime session bundling the PJRT API, client, and selected device.
data Session = Session
    { sessionApi     :: !PJRTApi
    , sessionClient  :: !PJRTClient
    , _sessionDevice :: !PJRTDevice
    }

-- | Bracket-style CPU session.
withCPU :: (Session -> IO a) -> IO a
withCPU action = do
    path <- getPluginPath "cpu" "libpjrt_cpu.so"
    withPJRT path $ \api client -> do
        devs <- Dev.addressableDevices api client
        case devs of
            [] -> error "PJRT CPU client has no addressable devices"
            (d:_) -> action (Session api client d)

-- | Bracket-style GPU session (first available GPU).
withGPU :: (Session -> IO a) -> IO a
withGPU action = withGPUDevice 0 action

-- | Bracket-style GPU session with explicit device index.
withGPUDevice :: Int -> (Session -> IO a) -> IO a
withGPUDevice idx action = do
    path <- getPluginPath "gpu" "libpjrt_cuda.so"
    withPJRT path $ \api client -> do
        devs <- Dev.addressableDevices api client
        let gpuDevs = filter (\d -> not (isCpuDevice api d)) devs
        when (null gpuDevs) $
            error "No GPU devices found"
        when (idx < 0 || idx >= length gpuDevs) $
            error $ "GPU device index " ++ show idx ++ " out of range ("
                 ++ show (length gpuDevs) ++ " GPUs available)"
        action (Session api client (gpuDevs !! idx))

-- | Construct a 'Session' from an existing PJRT API, client, and device.
-- This is useful when you already manage the plugin lifecycle externally
-- (e.g. in a test harness that shares one client across many tests).
sessionFrom :: PJRTApi -> PJRTClient -> PJRTDevice -> Session
sessionFrom = Session

isCpuDevice :: PJRTApi -> PJRTDevice -> Bool
isCpuDevice api dev = unsafePerformIO $ do
    kind <- Dev.deviceKind api dev
    return $ map (\c -> if c >= 'A' && c <= 'Z' then toEnum (fromEnum c + 32) else c) kind == "cpu"

-- ---------------------------------------------------------------------------
-- Compilation
-- ---------------------------------------------------------------------------

-- | An opaque compiled executable handle.
data Compiled = Compiled
    { _compiledSession :: !Session
    , compiledExec    :: !PJRTExecutable
    }

-- | Compile a StableHLO module for the session's device.
compile :: Session -> Module -> IO Compiled
compile sess modu = do
    exec <- compileWithOptions (sessionApi sess) (sessionClient sess)
                                (render modu) defaultCompileOptions
    return (Compiled sess exec)

-- ---------------------------------------------------------------------------
-- Host tensors
-- ---------------------------------------------------------------------------

-- | A typed host tensor carrying its shape and dtype as phantom types.
newtype HostTensor (s :: Shape) (d :: DType) = HostTensor
    { unHostTensor :: V.Vector (HostType d)
    }

-- | Construct a 'HostTensor' from a list. Fast path: no length validation.
hostFromList :: (KnownShape s, KnownDType d, V.Storable (HostType d)) => [HostType d] -> HostTensor s d
hostFromList = HostTensor . V.fromList

-- | Construct a 'HostTensor' from a vector. Fast path: no length validation.
hostFromVector :: (KnownShape s, KnownDType d, V.Storable (HostType d)) => V.Vector (HostType d) -> HostTensor s d
hostFromVector = HostTensor

-- | Expected number of elements for a shape.
expectedElems :: forall s. KnownShape s => Int
expectedElems = fromIntegral $ product (shapeVal (Proxy @s))

-- | Safe variant that validates length.
hostFromListSafe :: forall s d. (KnownShape s, KnownDType d, V.Storable (HostType d)) => [HostType d] -> Either String (HostTensor s d)
hostFromListSafe xs =
    let vec = V.fromList xs
    in hostFromVectorSafe vec

-- | Safe variant that validates length.
hostFromVectorSafe :: forall s d. (KnownShape s, KnownDType d, V.Storable (HostType d)) => V.Vector (HostType d) -> Either String (HostTensor s d)
hostFromVectorSafe vec
    | V.length vec == expectedElems @s = Right (HostTensor vec)
    | otherwise = Left $ unlines
        [ "HostTensor shape mismatch"
        , "  Expected elements: " ++ show (expectedElems @s)
        , "  Actual elements:   " ++ show (V.length vec)
        , "  Shape:             " ++ show (shapeVal (Proxy @s))
        ]

-- | Extract the underlying vector.
hostToVector :: HostTensor s d -> V.Vector (HostType d)
hostToVector = unHostTensor

-- | Extract as a list.
hostToList :: V.Storable (HostType d) => HostTensor s d -> [HostType d]
hostToList = V.toList . unHostTensor

-- ---------------------------------------------------------------------------
-- Device transfer
-- ---------------------------------------------------------------------------

-- | Map a 'DType' to its PJRT buffer type constant.
bufferTypeForDType :: DType -> CInt
bufferTypeForDType F32  = bufferTypeF32
bufferTypeForDType F64  = bufferTypeF64
bufferTypeForDType I8   = bufferTypeS8
bufferTypeForDType I16  = bufferTypeS16
bufferTypeForDType I32  = bufferTypeS32
bufferTypeForDType I64  = bufferTypeS64
bufferTypeForDType UI8  = bufferTypeU8
bufferTypeForDType UI16 = bufferTypeU16
bufferTypeForDType UI32 = bufferTypeU32
bufferTypeForDType UI64 = bufferTypeU64
bufferTypeForDType Bool = bufferTypePred
bufferTypeForDType dt   = error $ "bufferTypeForDType: unsupported dtype " ++ show dt

-- | Upload a typed vector to the device.
toDeviceTyped :: forall d. (KnownDType d, V.Storable (HostType d))
              => Session -> V.Vector (HostType d) -> [Int64] -> IO PJRTBuffer
toDeviceTyped sess vec dims =
    let dtype = bufferTypeForDType (dtypeVal (Proxy @d))
    in Buf.toDevice (sessionApi sess) (sessionClient sess) vec dims dtype

-- | Download a device buffer to a typed vector.
fromDeviceTyped :: forall d. (KnownDType d, V.Storable (HostType d))
                => Session -> PJRTBuffer -> Int -> IO (V.Vector (HostType d))
fromDeviceTyped sess buf n =
    Buf.fromDevice (sessionApi sess) buf n

-- ---------------------------------------------------------------------------
-- Typeclass machinery for multi-value I/O
-- ---------------------------------------------------------------------------

class ToDeviceInputs a where
    toInputs :: Session -> a -> IO [PJRTBuffer]
    inputCount :: Proxy a -> Int

instance (KnownShape s, KnownDType d, V.Storable (HostType d)) => ToDeviceInputs (HostTensor s d) where
    toInputs sess (HostTensor vec) = do
        let dims = map fromIntegral (shapeVal (Proxy @s))
        buf <- toDeviceTyped @d sess vec dims
        return [buf]
    inputCount _ = 1

instance (ToDeviceInputs a, ToDeviceInputs b) => ToDeviceInputs (a, b) where
    toInputs sess (a, b) = do
        bufsA <- toInputs sess a
        bufsB <- toInputs sess b
        return (bufsA ++ bufsB)
    inputCount _ = inputCount (Proxy @a) + inputCount (Proxy @b)

instance (ToDeviceInputs a, ToDeviceInputs b, ToDeviceInputs c) => ToDeviceInputs (a, b, c) where
    toInputs sess (a, b, c) = do
        bufsA <- toInputs sess a
        bufsB <- toInputs sess b
        bufsC <- toInputs sess c
        return (bufsA ++ bufsB ++ bufsC)
    inputCount _ = inputCount (Proxy @a) + inputCount (Proxy @b) + inputCount (Proxy @c)

class FromDeviceOutputs a where
    fromOutputs :: Session -> [PJRTBuffer] -> IO a
    outputCount :: Proxy a -> Int

instance (KnownShape s, KnownDType d, V.Storable (HostType d)) => FromDeviceOutputs (HostTensor s d) where
    fromOutputs sess [buf] = do
        let n = expectedElems @s
        vec <- fromDeviceTyped @d sess buf n
        return (HostTensor vec)
    fromOutputs _ _ = error "FromDeviceOutputs: expected exactly one buffer"
    outputCount _ = 1

instance (FromDeviceOutputs a, FromDeviceOutputs b) => FromDeviceOutputs (a, b) where
    fromOutputs sess bufs = do
        let (bufsA, bufsB) = splitAt (outputCount (Proxy @a)) bufs
        a <- fromOutputs sess bufsA
        b <- fromOutputs sess bufsB
        return (a, b)
    outputCount _ = outputCount (Proxy @a) + outputCount (Proxy @b)

instance (FromDeviceOutputs a, FromDeviceOutputs b, FromDeviceOutputs c) => FromDeviceOutputs (a, b, c) where
    fromOutputs sess bufs = do
        let (bufsA, rest) = splitAt (outputCount (Proxy @a)) bufs
            (bufsB, bufsC) = splitAt (outputCount (Proxy @b)) rest
        a <- fromOutputs sess bufsA
        b <- fromOutputs sess bufsB
        c <- fromOutputs sess bufsC
        return (a, b, c)
    outputCount _ = outputCount (Proxy @a) + outputCount (Proxy @b) + outputCount (Proxy @c)

-- ---------------------------------------------------------------------------
-- Execution
-- ---------------------------------------------------------------------------

-- | Synchronous execution: upload, run, download, and return.
run :: (ToDeviceInputs inputs, FromDeviceOutputs outputs)
    => Session -> Compiled -> inputs -> IO outputs
run sess compiled inputs = do
    inBufs <- toInputs sess inputs
    outBufs <- execute (sessionApi sess) (compiledExec compiled) inBufs
    fromOutputs sess outBufs

-- | Asynchronous execution: upload, launch, and return immediately.
--
-- NOTE: The current implementation downloads outputs synchronously before
-- returning. True non-blocking async (overlapping host work with device
-- execution) is planned and can be added without breaking this API.
runAsync :: (ToDeviceInputs inputs, FromDeviceOutputs outputs)
         => Session -> Compiled -> inputs -> IO outputs
runAsync = run

-- | Block until all asynchronous operations associated with the outputs
-- have completed. For the current implementation this is a no-op because
-- 'runAsync' already synchronizes before returning.
awaitOutputs :: FromDeviceOutputs outputs => Session -> outputs -> IO ()
awaitOutputs _ _ = return ()

-- ---------------------------------------------------------------------------
-- Dynamic-shape execution
-- ---------------------------------------------------------------------------

-- | A host tensor whose shape is only known at runtime.
data DynamicHostTensor d = DynamicHostTensor
    { dhtShape :: [Int64]          -- ^ runtime shape
    , dhtData  :: V.Vector (HostType d)
    }

-- | Construct a 'DynamicHostTensor' from a vector and explicit shape.
dynamicHostFromVector :: V.Vector (HostType d) -> [Int64] -> DynamicHostTensor d
dynamicHostFromVector vec shape = DynamicHostTensor shape vec

-- | Extract the underlying vector from a 'DynamicHostTensor'.
dynamicHostToVector :: DynamicHostTensor d -> V.Vector (HostType d)
dynamicHostToVector = dhtData

-- | Upload a dynamic host tensor to the device.
toDeviceDynamic :: forall d. (KnownDType d, V.Storable (HostType d))
                => Session -> DynamicHostTensor d -> IO PJRTBuffer
toDeviceDynamic sess (DynamicHostTensor shape vec) =
    let dtype = bufferTypeForDType (dtypeVal (Proxy @d))
    in Buf.toDevice (sessionApi sess) (sessionClient sess) vec shape dtype

-- | Download a device buffer to a dynamic host tensor.
fromDeviceDynamic :: forall d. (KnownDType d, V.Storable (HostType d))
                  => Session -> PJRTBuffer -> IO (DynamicHostTensor d)
fromDeviceDynamic sess buf = do
    dims <- Buf.bufferDimensions (sessionApi sess) buf
    let n = fromIntegral $ product dims
    vec <- fromDeviceTyped @d sess buf n
    return $ DynamicHostTensor dims vec

-- | Run a compiled module that accepts dynamically-shaped inputs.
-- Works on both CPU and GPU backends.
runDynamic :: forall d. (KnownDType d, V.Storable (HostType d))
           => Session -> Compiled -> [DynamicHostTensor d] -> IO [DynamicHostTensor d]
runDynamic sess compiled inputs = do
    inBufs <- mapM (toDeviceDynamic sess) inputs
    outBufs <- execute (sessionApi sess) (compiledExec compiled) inBufs
    mapM (fromDeviceDynamic sess) outBufs

-- | Asynchronous variant of 'runDynamic'.
runDynamicAsync :: forall d. (KnownDType d, V.Storable (HostType d))
                => Session -> Compiled -> [DynamicHostTensor d] -> IO [DynamicHostTensor d]
runDynamicAsync = runDynamic
