{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# LANGUAGE OverloadedStrings #-}

module HHLO.Core.Types
    ( DType(..)
    , Shape
    , Dim
    , KnownShape(..)
    , dtypeToText
    , HostType
    ) where

import GHC.TypeLits
import Data.Proxy
import Data.Kind (Type)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Word (Word8, Word16, Word32, Word64)
import Data.Text (Text)


-- | Supported element types for tensors.
data DType
    = F32 | F64
    | I8 | I16 | I32 | I64
    | UI8 | UI16 | UI32 | UI64
    | Bool
    | Complex DType
    deriving (Eq, Show, Ord)

-- | A dimension is a type-level natural number.
type Dim = Nat

-- | A shape is a type-level list of dimensions.
type Shape = [Nat]

-- | Obtain the runtime value of a type-level shape.
class KnownShape (s :: Shape) where
    shapeVal :: Proxy s -> [Integer]

instance KnownShape '[] where
    shapeVal _ = []

instance (KnownNat n, KnownShape ns) => KnownShape (n ': ns) where
    shapeVal _ = natVal (Proxy @n) : shapeVal (Proxy @ns)

-- | Convert a 'DType' to its MLIR textual representation.
dtypeToText :: DType -> Text
dtypeToText F32     = "f32"
dtypeToText F64     = "f64"
dtypeToText I32     = "i32"
dtypeToText I64     = "i64"
dtypeToText I8      = "i8"
dtypeToText I16     = "i16"
dtypeToText UI8     = "ui8"
dtypeToText UI16    = "ui16"
dtypeToText UI32    = "ui32"
dtypeToText UI64    = "ui64"
dtypeToText Bool    = "i1"
dtypeToText (Complex F32) = "complex<f32>"
dtypeToText (Complex F64) = "complex<f64>"
dtypeToText dt      = error $ "Unsupported dtype: " ++ show dt

-- | Mapping from 'DType' to the corresponding Haskell host type.
-- 'Bool' maps to 'Word8' because PJRT's PRED buffer type transfers
-- as single-byte boolean values.
type family HostType (d :: DType) :: Type where
    HostType 'F32  = Float
    HostType 'F64  = Double
    HostType 'I8   = Int8
    HostType 'I16  = Int16
    HostType 'I32  = Int32
    HostType 'I64  = Int64
    HostType 'UI8  = Word8
    HostType 'UI16 = Word16
    HostType 'UI32 = Word32
    HostType 'UI64 = Word64
    HostType 'Bool = Word8
