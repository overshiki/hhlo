{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.EDSL.Ops
    ( add
    , sub
    , multiply
    , divide
    , matmul
    , linear
    , linearBatched
    -- * Unary element-wise ops
    , relu
    , negate
    , abs'
    , exponential
    , logarithm
    -- * Binary element-wise ops
    , maximum
    , minimum
    -- * Shape manipulation
    , reshape
    , broadcastWithDims
    , transpose
    -- * Reductions
    , reduceSum
    -- * Constants
    , constant
    -- * Tuple
    , Tuple2(..)
    , returnTuple2
    , Tuple(..)
    , returnT
    ) where

import Prelude hiding (subtract, negate, maximum, minimum, abs)

import Data.Int (Int64)
import Data.Proxy
import GHC.TypeLits
import HHLO.Core.Types
import HHLO.IR.AST
import HHLO.IR.Builder

-- ---------------------------------------------------------------------------
-- Binary element-wise ops
-- ---------------------------------------------------------------------------

add :: forall s1 s2 d1 d2. (s1 ~ s2, d1 ~ d2, KnownShape s1, KnownDType d1)
    => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)
add (Tensor x) (Tensor y) = do
    let ttype = tensorType (Proxy @s1) (Proxy @d1)
    vid <- emitOp "stablehlo.add" [x, y] [ttype, ttype] [] ttype
    return (Tensor vid)

sub :: forall s1 s2 d1 d2. (s1 ~ s2, d1 ~ d2, KnownShape s1, KnownDType d1)
    => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)
sub (Tensor x) (Tensor y) = do
    let ttype = tensorType (Proxy @s1) (Proxy @d1)
    vid <- emitOp "stablehlo.subtract" [x, y] [ttype, ttype] [] ttype
    return (Tensor vid)

multiply :: forall s1 s2 d1 d2. (s1 ~ s2, d1 ~ d2, KnownShape s1, KnownDType d1)
         => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)
multiply (Tensor x) (Tensor y) = do
    let ttype = tensorType (Proxy @s1) (Proxy @d1)
    vid <- emitOp "stablehlo.multiply" [x, y] [ttype, ttype] [] ttype
    return (Tensor vid)

divide :: forall s1 s2 d1 d2. (s1 ~ s2, d1 ~ d2, KnownShape s1, KnownDType d1)
       => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)
divide (Tensor x) (Tensor y) = do
    let ttype = tensorType (Proxy @s1) (Proxy @d1)
    vid <- emitOp "stablehlo.divide" [x, y] [ttype, ttype] [] ttype
    return (Tensor vid)

maximum :: forall s1 s2 d1 d2. (s1 ~ s2, d1 ~ d2, KnownShape s1, KnownDType d1)
        => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)
maximum (Tensor x) (Tensor y) = do
    let ttype = tensorType (Proxy @s1) (Proxy @d1)
    vid <- emitOp "stablehlo.maximum" [x, y] [ttype, ttype] [] ttype
    return (Tensor vid)

minimum :: forall s1 s2 d1 d2. (s1 ~ s2, d1 ~ d2, KnownShape s1, KnownDType d1)
        => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)
minimum (Tensor x) (Tensor y) = do
    let ttype = tensorType (Proxy @s1) (Proxy @d1)
    vid <- emitOp "stablehlo.minimum" [x, y] [ttype, ttype] [] ttype
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Matrix multiplication
-- ---------------------------------------------------------------------------

type family MatMulShape (a :: Shape) (b :: Shape) :: Shape where
    MatMulShape '[m, k] '[k, n] = '[m, n]
    MatMulShape '[k]    '[k, n] = '[n]
    MatMulShape '[m, k] '[k]    = '[m]

matmul :: forall s1 s2 d. (KnownShape s1, KnownShape s2, KnownShape (MatMulShape s1 s2), KnownDType d)
       => Tensor s1 d -> Tensor s2 d -> Builder (Tensor (MatMulShape s1 s2) d)
matmul (Tensor x) (Tensor y) = do
    let inType1 = tensorType (Proxy @s1) (Proxy @d)
        inType2 = tensorType (Proxy @s2) (Proxy @d)
        outType = tensorType (Proxy @(MatMulShape s1 s2)) (Proxy @d)
    vid <- emitOp "stablehlo.dot" [x, y] [inType1, inType2] [] outType
    return (Tensor vid)

-- | A linear (fully-connected) layer: @matmul x w + b@.
-- For a single sample (no batch dimension):
--   * @x@ has shape @[in_features]@
--   * @w@ has shape @[in_features, out_features]@
--   * @b@ has shape @[out_features]@
linear :: ( KnownShape s1, KnownShape s2, KnownShape (MatMulShape s1 s2)
          , KnownDType d, s3 ~ MatMulShape s1 s2 )
       => Tensor s1 d -> Tensor s2 d -> Tensor s3 d -> Builder (Tensor s3 d)
linear x w b = do
    y <- matmul x w
    add y b

-- | Batched linear layer: @matmul x w + broadcast(bias)@.
--
-- * @x@ has shape @[batch, in_features]@
-- * @w@ has shape @[in_features, out_features]@
-- * @b@ has shape @[out_features]@
-- * Result has shape @[batch, out_features]@
linearBatched :: forall batch inDim outDim d.
                 ( KnownNat batch, KnownNat inDim, KnownNat outDim, KnownDType d )
              => Tensor '[batch, inDim] d -> Tensor '[inDim, outDim] d -> Tensor '[outDim] d
              -> Builder (Tensor '[batch, outDim] d)
linearBatched x w b = do
    y <- matmul x w
    b' <- broadcastWithDims @'[outDim] @'[batch, outDim] [1] b
    add y b'

-- ---------------------------------------------------------------------------
-- Unary element-wise ops
-- ---------------------------------------------------------------------------

-- | Rectified Linear Unit: max(x, 0).
relu :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
relu t = do
    zero <- constant @s @d 0.0
    maximum t zero

negate :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
negate (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.negate" [x] [ttype] [] ttype
    return (Tensor vid)

abs' :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
abs' (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.abs" [x] [ttype] [] ttype
    return (Tensor vid)

exponential :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
exponential (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.exponential" [x] [ttype] [] ttype
    return (Tensor vid)

logarithm :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
logarithm (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.log" [x] [ttype] [] ttype
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Shape manipulation
-- ---------------------------------------------------------------------------

-- | Reshape a tensor to a new shape.
-- The caller must ensure the total element count matches.
reshape :: forall sFrom sTo d. (KnownShape sFrom, KnownShape sTo, KnownDType d)
        => Tensor sFrom d -> Builder (Tensor sTo d)
reshape (Tensor x) = do
    let inType = tensorType (Proxy @sFrom) (Proxy @d)
        outType = tensorType (Proxy @sTo) (Proxy @d)
    vid <- emitOp "stablehlo.reshape" [x] [inType] [] outType
    return (Tensor vid)

-- | Broadcast a tensor to a new shape with explicit dimension mapping.
--
-- Example: broadcast a bias @[8] to @[batch, 8] with dims @[1]:
-- @
--   b' <- broadcastWithDims @'[8] @'[batch, 8] [1] b
-- @
--
-- The 'dims' list maps each dimension of the input to a dimension in the
-- output.  See the StableHLO 'broadcast_in_dim' spec for details.
broadcastWithDims :: forall sFrom sTo d. (KnownShape sFrom, KnownShape sTo, KnownDType d)
                  => [Int64] -> Tensor sFrom d -> Builder (Tensor sTo d)
broadcastWithDims dims (Tensor x) = do
    let inType = tensorType (Proxy @sFrom) (Proxy @d)
        outType = tensorType (Proxy @sTo) (Proxy @d)
    vid <- emitOp "stablehlo.broadcast_in_dim" [x] [inType]
        [AttrIntList "dims" (map fromIntegral dims)] outType
    return (Tensor vid)

-- | Transpose a tensor by permuting dimensions.
transpose :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
transpose (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.transpose" [x] [ttype] [] ttype
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Reductions
-- ---------------------------------------------------------------------------

type family ReduceAllShape (s :: Shape) :: Shape where
    ReduceAllShape '[]       = '[]
    ReduceAllShape (_ ': xs) = ReduceAllShape xs

-- | Sum all elements of a tensor (reduce over all dimensions).
-- Result is a scalar.
reduceSum :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor '[] d)
reduceSum (Tensor x) = do
    let inType = tensorType (Proxy @s) (Proxy @d)
        outType = tensorType (Proxy @'[]) (Proxy @d)
    zeroVid <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [] (dtypeVal (Proxy @d)) [0.0]] outType
    let dims = [0 .. fromIntegral (length (shapeVal (Proxy @s))) - 1]
    vid <- emitReduce x inType zeroVid outType dims "stablehlo.add" outType
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Constants
-- ---------------------------------------------------------------------------

-- | Create a constant tensor filled with a single value.
constant :: forall s d. (KnownShape s, KnownDType d) => Double -> Builder (Tensor s d)
constant val = do
    let outType = tensorType (Proxy @s) (Proxy @d)
        shp = shapeVal (Proxy @s)
        numElems = product shp
    vid <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements shp (dtypeVal (Proxy @d)) (replicate (fromIntegral numElems) val)] outType
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Tuple return
-- ---------------------------------------------------------------------------

-- | Return a pair of tensors from a multi-result builder.
returnTuple2 :: Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tuple2 s1 d1 s2 d2)
returnTuple2 t1 t2 = return (Tuple2 t1 t2)

-- | Return a heterogeneous tuple of tensors from a multi-result builder.
returnT :: Tuple ss ds -> Builder (Tuple ss ds)
returnT = return
