{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# LANGUAGE OverloadedStrings #-}

module HHLO.Core.Types
    ( DType(..)
    , Shape
    , Dim
    , KnownShape(..)
    , dtypeToText
    , HostType
    -- * Fixed-length configuration vectors
    , Length
    , V
    , V1
    , V2
    , V3
    , V4
    , Padding
    , P2
    , v1
    , v2
    , v3
    , v4
    , p2
    ) where

import GHC.TypeLits
import Data.Proxy
import Data.Kind (Type)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Word (Word8, Word16, Word32, Word64)
import Data.Text (Text)
import qualified Data.Vector.Sized as VS


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

-- ---------------------------------------------------------------------------
-- Fixed-length configuration vectors
-- ---------------------------------------------------------------------------

-- | Compute the length of a type-level list.
type family Length (xs :: [k]) :: Nat where
    Length '[]     = 0
    Length (x:xs)  = 1 + Length xs

-- | Fixed-length vector alias.
type V (n :: Nat) a = VS.Vector n a

-- | 1-element vector.
type V1 a = V 1 a

-- | 2-element vector (e.g. spatial height, width).
type V2 a = V 2 a

-- | 3-element vector.
type V3 a = V 3 a

-- | 4-element vector (e.g. NHWC dimensions).
type V4 a = V 4 a

-- | Padding config: one (low,high) pair per dimension.
type Padding (n :: Nat) = V n (Int64, Int64)

-- | 2D padding alias (common case).
type P2 = Padding 2

-- | Smart constructor for a 1-element vector.
v1 :: a -> V1 a
v1 a = a `VS.cons` VS.empty

-- | Smart constructor for a 2-element vector.
v2 :: a -> a -> V2 a
v2 a b = a `VS.cons` (b `VS.cons` VS.empty)

-- | Smart constructor for a 3-element vector.
v3 :: a -> a -> a -> V3 a
v3 a b c = a `VS.cons` (b `VS.cons` (c `VS.cons` VS.empty))

-- | Smart constructor for a 4-element vector.
v4 :: a -> a -> a -> a -> V4 a
v4 a b c d = a `VS.cons` (b `VS.cons` (c `VS.cons` (d `VS.cons` VS.empty)))

-- | Smart constructor for 2D padding from two (before,after) pairs.
p2 :: (Int64, Int64) -> (Int64, Int64) -> P2
p2 = v2
