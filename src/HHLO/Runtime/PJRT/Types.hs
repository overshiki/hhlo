{-# LANGUAGE ForeignFunctionInterface #-}

module HHLO.Runtime.PJRT.Types
    ( PJRTApi(..)
    , PJRTClient(..)
    , PJRTBuffer(..)
    , PJRTExecutable(..)
    , PJRTError(..)
    , PJRTEvent(..)
    , PJRTDevice(..)
    -- * ForeignPtr lifetime helpers
    , withBufferPtr
    , withExecPtr
    , withBufferPtrs
    -- * Buffer type constants
    , bufferTypeInvalid
    , bufferTypePred
    , bufferTypeS8
    , bufferTypeS16
    , bufferTypeS32
    , bufferTypeS64
    , bufferTypeU8
    , bufferTypeU16
    , bufferTypeU32
    , bufferTypeU64
    , bufferTypeF16
    , bufferTypeF32
    , bufferTypeF64
    , bufferTypeBF16
    , bufferTypeC64
    , bufferTypeC128
    ) where

import Foreign.C.Types (CInt(..))
import Foreign.ForeignPtr (ForeignPtr, touchForeignPtr)
import GHC.ForeignPtr (unsafeForeignPtrToPtr)
import Foreign.Ptr (Ptr)
import System.IO.Unsafe (unsafePerformIO)

newtype PJRTApi        = PJRTApi        (Ptr PJRTApi)
newtype PJRTClient     = PJRTClient     (Ptr PJRTClient)
newtype PJRTBuffer     = PJRTBuffer     (ForeignPtr PJRTBuffer)
newtype PJRTExecutable = PJRTExecutable (ForeignPtr PJRTExecutable)
newtype PJRTError      = PJRTError      (Ptr PJRTError)
newtype PJRTEvent      = PJRTEvent      (Ptr PJRTEvent)
newtype PJRTDevice     = PJRTDevice     (Ptr PJRTDevice)

-- | Keep a 'PJRTBuffer' alive across an FFI call.
--
-- This is the standard @unsafeForeignPtrToPtr + touchForeignPtr@
-- bracket pattern.  It prevents GHC from collecting (and finalizing)
-- the buffer while the C function is still using it.
withBufferPtr :: PJRTBuffer -> (Ptr PJRTBuffer -> IO a) -> IO a
withBufferPtr (PJRTBuffer fp) action = do
    r <- action (unsafeForeignPtrToPtr fp)
    touchForeignPtr fp
    return r

-- | Keep a 'PJRTExecutable' alive across an FFI call.
withExecPtr :: PJRTExecutable -> (Ptr PJRTExecutable -> IO a) -> IO a
withExecPtr (PJRTExecutable fp) action = do
    r <- action (unsafeForeignPtrToPtr fp)
    touchForeignPtr fp
    return r

-- | Keep a list of 'PJRTBuffer's alive across an FFI call.
--
-- The action receives the raw 'Ptr' values; after it returns every
-- buffer is touched so that GHC does not run their finalizers
-- prematurely.
withBufferPtrs :: [PJRTBuffer] -> ([Ptr PJRTBuffer] -> IO a) -> IO a
withBufferPtrs buffers action = do
    let ptrs = map (\(PJRTBuffer fp) -> unsafeForeignPtrToPtr fp) buffers
    r <- action ptrs
    mapM_ (\(PJRTBuffer fp) -> touchForeignPtr fp) buffers
    return r

-- ---------------------------------------------------------------------------
-- Buffer type constants (fetched from C shim at first use)
-- ---------------------------------------------------------------------------

foreign import ccall "pjrt_shim.h hhlo_buffer_type_invalid" c_bufferTypeInvalid :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_pred"    c_bufferTypePred    :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_s8"      c_bufferTypeS8      :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_s16"     c_bufferTypeS16     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_s32"     c_bufferTypeS32     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_s64"     c_bufferTypeS64     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_u8"      c_bufferTypeU8      :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_u16"     c_bufferTypeU16     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_u32"     c_bufferTypeU32     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_u64"     c_bufferTypeU64     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_f16"     c_bufferTypeF16     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_f32"     c_bufferTypeF32     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_f64"     c_bufferTypeF64     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_bf16"    c_bufferTypeBF16    :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_c64"     c_bufferTypeC64     :: IO CInt
foreign import ccall "pjrt_shim.h hhlo_buffer_type_c128"    c_bufferTypeC128    :: IO CInt

bufType :: IO CInt -> CInt
bufType action = unsafePerformIO action
{-# NOINLINE bufType #-}

bufferTypeInvalid :: CInt
bufferTypeInvalid = bufType c_bufferTypeInvalid

bufferTypePred :: CInt
bufferTypePred = bufType c_bufferTypePred

bufferTypeS8 :: CInt
bufferTypeS8 = bufType c_bufferTypeS8

bufferTypeS16 :: CInt
bufferTypeS16 = bufType c_bufferTypeS16

bufferTypeS32 :: CInt
bufferTypeS32 = bufType c_bufferTypeS32

bufferTypeS64 :: CInt
bufferTypeS64 = bufType c_bufferTypeS64

bufferTypeU8 :: CInt
bufferTypeU8 = bufType c_bufferTypeU8

bufferTypeU16 :: CInt
bufferTypeU16 = bufType c_bufferTypeU16

bufferTypeU32 :: CInt
bufferTypeU32 = bufType c_bufferTypeU32

bufferTypeU64 :: CInt
bufferTypeU64 = bufType c_bufferTypeU64

bufferTypeF16 :: CInt
bufferTypeF16 = bufType c_bufferTypeF16

bufferTypeF32 :: CInt
bufferTypeF32 = bufType c_bufferTypeF32

bufferTypeF64 :: CInt
bufferTypeF64 = bufType c_bufferTypeF64

bufferTypeBF16 :: CInt
bufferTypeBF16 = bufType c_bufferTypeBF16

bufferTypeC64 :: CInt
bufferTypeC64 = bufType c_bufferTypeC64

bufferTypeC128 :: CInt
bufferTypeC128 = bufType c_bufferTypeC128
