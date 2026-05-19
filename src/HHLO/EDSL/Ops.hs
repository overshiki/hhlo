{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.EDSL.Ops
    ( add
    , sub
    , multiply
    , divide
    , matmul
    , dotGeneral
    , einsum
    , linear
    , linearBatched
    -- * Unary element-wise ops
    , relu
    , negate
    , abs'
    , exponential
    , logarithm
    , tanh
    , erf
    , sqrt
    , rsqrt
    , sin
    , cos
    , tan
    , log1p
    , floor
    , ceil
    -- * Binary element-wise ops
    , maximum
    , minimum
    , pow
    -- * Shape manipulation
    , reshape
    , broadcastWithDims
    , transpose
    , concatenate
    , concatenate2
    , iota
    , split
    , stack
    -- * Reductions
    , reduceSum
    , reduceSumDim
    , productAll
    , productDim
    , reduceWindow
    , maxPool
    , avgPool
    -- * Neural network layers
    , softmax1D
    , softmax2D
    , softmax3D
    , softmax4D
    , conv2d
    , conv2dWithPadding
    , transposeConvolution
    , batchNormInference
    -- * Fixed-length configuration helpers
    , Length
    , V
    , V1
    , V2
    , V3
    , V4
    , Padding
    , v1
    , v2
    , v3
    , v4
    , p2
    , layerNorm
    , globalAvgPool
    , gelu
    -- * Control flow
    , whileLoop
    , whileLoopN
    , whileLoop2
    , whileLoop3
    , whileLoop4
    , whileLoop5
    , whileLoop6
    , whileLoop7
    , whileLoop8
    , conditional
    , conditional2
    , conditional3
    , conditional4
    , conditional5
    , conditional6
    , conditional7
    , conditional8
    , compare
    , lessThan
    , greaterThan
    , equal
    , notEqual
    , lessThanOrEqual
    , greaterThanOrEqual
    , logicalAnd
    , logicalOr
    , logicalNot
    -- * Data movement
    , gather
    , scatter
    , slice
    , pad
    , dynamicSlice
    , sort
    , convert
    , topK
    -- * Selection
    , select
    -- * Map
    , map
    -- * Constants
    , constant
    -- * Tuple
    , Tuple2(..)
    , Tuple3(..)
    , Tuple4(..)
    , Tuple5(..)
    , Tuple6(..)
    , Tuple7(..)
    , Tuple8(..)
    , returnTuple2
    , returnTuple3
    , returnTuple4
    , returnTuple5
    , returnTuple6
    , returnTuple7
    , returnTuple8
    , Tuple(..)
    , returnT
    -- * Random number generation
    , rngUniform
    , rngNormal
    , rngBitGenerator
    -- * Composite / convenience
    , sigmoid
    , sumAll
    , pack2
    , pack3
    , slice1
    -- * Custom calls
    , customCall1
    , customCall2
    , customCallRaw
    , customVJP1
    ) where

import Prelude hiding (subtract, negate, maximum, minimum, abs, compare, map, tanh, sqrt, sin, cos, tan, floor, ceiling)

import Control.Monad (when)
import Data.Int (Int64, Int32)
import Data.List (elemIndex)
import Data.Maybe (fromJust)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import GHC.TypeLits
import HHLO.Core.Types
import HHLO.IR.AST
import HHLO.IR.Builder
import qualified Data.Vector.Sized as VS
import Data.Vector.Sized (Vector)
import qualified Data.Vector as V
import Unsafe.Coerce (unsafeCoerce)

-- | Unsafe helper: convert a list to a fixed-length vector.
-- The caller must ensure the list length equals the rank of shape @s@.
vecFromListUnsafe :: forall s a. KnownShape s => [a] -> Vector (Length s) a
vecFromListUnsafe xs =
    let expected = length (shapeVal (Proxy @s))
        actual   = length xs
    in if expected == actual
       then unsafeCoerce (V.fromList xs)
       else error $ "vecFromListUnsafe: expected length " ++ show expected ++ ", got " ++ show actual

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

pow :: forall s1 s2 d1 d2. (s1 ~ s2, d1 ~ d2, KnownShape s1, KnownDType d1)
    => Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor s1 d1)
pow (Tensor x) (Tensor y) = do
    let ttype = tensorType (Proxy @s1) (Proxy @d1)
    vid <- emitOp "stablehlo.power" [x, y] [ttype, ttype] [] ttype
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Matrix multiplication
-- ---------------------------------------------------------------------------

type family MatMulShape (a :: Shape) (b :: Shape) :: Shape where
    MatMulShape '[m, k] '[k, n] = '[m, n]
    MatMulShape '[k]    '[k, n] = '[n]
    MatMulShape '[m, k] '[k]    = '[m]
    MatMulShape '[batch, m, k] '[k, n] = '[batch, m, n]
    MatMulShape '[b, m, k] '[b, k, n] = '[b, m, n]
    MatMulShape '[b1, b2, m, k] '[b1, b2, k, n] = '[b1, b2, m, n]

matmul :: forall s1 s2 d. (KnownShape s1, KnownShape s2, KnownShape (MatMulShape s1 s2), KnownDType d)
       => Tensor s1 d -> Tensor s2 d -> Builder (Tensor (MatMulShape s1 s2) d)
matmul (Tensor x) (Tensor y) = do
    let inType1 = tensorType (Proxy @s1) (Proxy @d)
        inType2 = tensorType (Proxy @s2) (Proxy @d)
        outType = tensorType (Proxy @(MatMulShape s1 s2)) (Proxy @d)
        s1Shape = shapeVal (Proxy @s1)
        s2Shape = shapeVal (Proxy @s2)
        rank1 = length s1Shape
        rank2 = length s2Shape
        (batchL, batchR, contractL, contractR) = case (rank1, rank2) of
            (1, 2) -> ([], [], [0], [0])
            (2, 1) -> ([], [], [1], [0])
            (2, 2) -> ([], [], [1], [0])
            (3, 2) -> ([], [], [2], [0])
            (3, 3) -> ([0], [0], [2], [1])
            (4, 4) -> ([0,1], [0,1], [3], [2])
            _ -> error $ "matmul: unsupported ranks " ++ show rank1 ++ "x" ++ show rank2
        attrs =
            [ AttrIntList "lhs_batching_dimensions" (fmap fromIntegral batchL)
            , AttrIntList "rhs_batching_dimensions" (fmap fromIntegral batchR)
            , AttrIntList "lhs_contracting_dimensions" (fmap fromIntegral contractL)
            , AttrIntList "rhs_contracting_dimensions" (fmap fromIntegral contractR)
            ]
    vid <- emitOp "stablehlo.dot_general" [x, y] [inType1, inType2] attrs outType
    return (Tensor vid)

-- | General dot product with explicit batch and contracting dimensions.
-- Use this for batched matrix multiplication (rank > 2).
dotGeneral :: forall s1 s2 sOut d nBatch nContract.
              (KnownShape s1, KnownShape s2, KnownShape sOut, KnownDType d, KnownNat nBatch, KnownNat nContract)
           => Vector nBatch Int64     -- ^ lhs batch dims
           -> Vector nBatch Int64     -- ^ rhs batch dims
           -> Vector nContract Int64  -- ^ lhs contracting dims
           -> Vector nContract Int64  -- ^ rhs contracting dims
           -> Tensor s1 d
           -> Tensor s2 d
           -> Builder (Tensor sOut d)
dotGeneral lhsBatch rhsBatch lhsContract rhsContract (Tensor x) (Tensor y) = do
    let inType1 = tensorType (Proxy @s1) (Proxy @d)
        inType2 = tensorType (Proxy @s2) (Proxy @d)
        outType = tensorType (Proxy @sOut) (Proxy @d)
        batchL    = AttrIntList "lhs_batching_dimensions" (VS.toList lhsBatch)
        batchR    = AttrIntList "rhs_batching_dimensions" (VS.toList rhsBatch)
        contractL = AttrIntList "lhs_contracting_dimensions" (VS.toList lhsContract)
        contractR = AttrIntList "rhs_contracting_dimensions" (VS.toList rhsContract)
    vid <- emitOp "stablehlo.dot_general" [x, y] [inType1, inType2]
            [ batchL, batchR, contractL, contractR
            ] outType
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
        [AttrIntList "broadcast_dimensions" (fromIntegral <$> dims)] outType
    return (Tensor vid)

-- | Transpose a tensor by permuting dimensions.
--
-- @perm@ must be a permutation of @[0 .. rank-1]@.
transpose :: forall sIn sOut d. (KnownShape sIn, KnownShape sOut, KnownDType d)
          => Vector (Length sIn) Int64 -> Tensor sIn d -> Builder (Tensor sOut d)
transpose perm (Tensor x) = do
    let inType  = tensorType (Proxy @sIn)  (Proxy @d)
        outType = tensorType (Proxy @sOut) (Proxy @d)
        permAttr = AttrIntList "permutation" (VS.toList perm)
    vid <- emitOp "stablehlo.transpose" [x] [inType] [permAttr] outType
    return (Tensor vid)

tanh :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
tanh (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.tanh" [x] [ttype] [] ttype
    return (Tensor vid)

erf :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
erf (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.erf" [x] [ttype] [] ttype
    return (Tensor vid)

sqrt :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
sqrt (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.sqrt" [x] [ttype] [] ttype
    return (Tensor vid)

rsqrt :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
rsqrt (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.rsqrt" [x] [ttype] [] ttype
    return (Tensor vid)

sin :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
sin (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.sine" [x] [ttype] [] ttype
    return (Tensor vid)

cos :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
cos (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.cosine" [x] [ttype] [] ttype
    return (Tensor vid)

tan :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
tan (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.tangent" [x] [ttype] [] ttype
    return (Tensor vid)

log1p :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
log1p (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.log_plus_one" [x] [ttype] [] ttype
    return (Tensor vid)

floor :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
floor (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.floor" [x] [ttype] [] ttype
    return (Tensor vid)

ceil :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor s d)
ceil (Tensor x) = do
    let ttype = tensorType (Proxy @s) (Proxy @d)
    vid <- emitOp "stablehlo.ceil" [x] [ttype] [] ttype
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
    let dims = [0 .. fromIntegral (length (shapeVal (Proxy @s))) - 1]
    reduceSumDim @s @'[] dims (Tensor x)

-- | Sum elements over specific dimensions.
--
-- The caller must specify the output shape via type applications.
-- Example: reduce a [batch, classes] tensor over dim 1 to get [batch]:
-- @
--   sumClasses <- reduceSumDim @'[batch, classes] @'[batch] [1] t
-- @
reduceSumDim :: forall sFrom sTo d.
                (KnownShape sFrom, KnownShape sTo, KnownDType d)
             => [Int] -> Tensor sFrom d -> Builder (Tensor sTo d)
reduceSumDim dims (Tensor x) = do
    let inType   = tensorType (Proxy @sFrom) (Proxy @d)
        outType  = tensorType (Proxy @sTo)   (Proxy @d)
        elemType = tensorType (Proxy @'[])   (Proxy @d)
    -- Init value for stablehlo.reduce (must be scalar)
    zeroVid <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [] (dtypeVal (Proxy @d)) [0.0]] elemType
    -- Build reduction region: two scalar args, apply stablehlo.add
    redBlock <- runBlockBuilder [elemType, elemType] $ do
        a <- arg @'[] @d
        b <- arg @'[] @d
        sumVid <- emitOp "stablehlo.add"
                    [tensorValue a, tensorValue b]
                    [elemType, elemType] [] elemType
        emitReturn [sumVid] [elemType]
    vid <- emitOpRegions "stablehlo.reduce"
            [x, zeroVid]
            [inType, elemType]
            [AttrIntList "dimensions" (fromIntegral <$> dims)]
            [Region [redBlock]]
            outType
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Neural network layers
-- ---------------------------------------------------------------------------

-- | Softmax over a 1-D tensor (no batch dimension).
--
-- @softmax(x)_i = exp(x_i) / sum_j(exp(x_j))@
softmax1D :: forall n. KnownNat n => Tensor '[n] 'F32 -> Builder (Tensor '[n] 'F32)
softmax1D x = do
    ex  <- exponential x
    sm  <- reduceSum @'[n] @'F32 ex
    sm' <- broadcastWithDims @'[] @'[n] [] sm
    divide ex sm'

-- | Softmax over the last dimension of a 2-D tensor (batched).
--
-- Input shape @[batch, classes]@; each row is independently normalized.
softmax2D :: forall batch classes.
             (KnownNat batch, KnownNat classes)
          => Tensor '[batch, classes] 'F32 -> Builder (Tensor '[batch, classes] 'F32)
softmax2D x = do
    ex  <- exponential x
    sm  <- reduceSumDim @'[batch, classes] @'[batch] [1] ex
    sm' <- broadcastWithDims @'[batch] @'[batch, classes] [0] sm
    divide ex sm'

-- | Softmax over the last dimension of a 3-D tensor.
softmax3D :: forall a b c.
             (KnownNat a, KnownNat b, KnownNat c)
          => Tensor '[a, b, c] 'F32 -> Builder (Tensor '[a, b, c] 'F32)
softmax3D x = do
    ex  <- exponential x
    sm  <- reduceSumDim @'[a, b, c] @'[a, b] [2] ex
    sm' <- broadcastWithDims @'[a, b] @'[a, b, c] [0, 1] sm
    divide ex sm'

-- | Softmax over the last dimension of a 4-D tensor.
softmax4D :: forall a b c d.
             (KnownNat a, KnownNat b, KnownNat c, KnownNat d)
          => Tensor '[a, b, c, d] 'F32 -> Builder (Tensor '[a, b, c, d] 'F32)
softmax4D x = do
    ex  <- exponential x
    sm  <- reduceSumDim @'[a, b, c, d] @'[a, b, c] [3] ex
    sm' <- broadcastWithDims @'[a, b, c] @'[a, b, c, d] [0, 1, 2] sm
    divide ex sm'

-- | 2-D convolution (NHWC format).
--
-- * @input@ has shape @[batch, h, w, in_channels]@
-- * @kernel@ has shape @[kernel_h, kernel_w, in_channels, out_channels]@
-- * Result has shape @[batch, out_h, out_w, out_channels]@
--
-- Default settings: stride=1, padding=0, dilation=1, groups=1.
conv2d :: forall batch h w inCh outCh kh kw oh ow.
          ( KnownNat batch, KnownNat h, KnownNat w, KnownNat inCh, KnownNat outCh
          , KnownNat kh, KnownNat kw, KnownNat oh, KnownNat ow )
       => Tensor '[batch, h, w, inCh] 'F32
       -> Tensor '[kh, kw, inCh, outCh] 'F32
       -> Builder (Tensor '[batch, oh, ow, outCh] 'F32)
conv2d x k = conv2dWithPadding @batch @h @w @inCh @outCh @kh @kw @oh @ow (v2 1 1) (p2 (0, 0) (0, 0)) x k

-- | 2-D convolution with explicit stride and padding (NHWC format).
--
-- * @input@ has shape @[batch, h, w, in_channels]@
-- * @kernel@ has shape @[kernel_h, kernel_w, in_channels, out_channels]@
-- * @strides@ is @[stride_h, stride_w]@
-- * @padding@ is @[[pad_top, pad_bottom], [pad_left, pad_right]]@
-- * Result has shape @[batch, out_h, out_w, out_channels]@
conv2dWithPadding :: forall batch h w inCh outCh kh kw oh ow.
                     ( KnownNat batch, KnownNat h, KnownNat w, KnownNat inCh, KnownNat outCh
                     , KnownNat kh, KnownNat kw, KnownNat oh, KnownNat ow )
                  => V2 Int64    -- ^ strides [sh, sw]
                  -> P2          -- ^ padding [[pt, pb], [pl, pr]]
                  -> Tensor '[batch, h, w, inCh] 'F32
                  -> Tensor '[kh, kw, inCh, outCh] 'F32
                  -> Builder (Tensor '[batch, oh, ow, outCh] 'F32)
conv2dWithPadding strides padding input kernel = do
    let inType1 = tensorType (Proxy @'[batch, h, w, inCh])   (Proxy @'F32)
        inType2 = tensorType (Proxy @'[kh, kw, inCh, outCh]) (Proxy @'F32)
        outType = tensorType (Proxy @'[batch, oh, ow, outCh]) (Proxy @'F32)
        strideVals = VS.toList strides
        padVals    = concatMap (\(l, h) -> [l, h]) (VS.toList padding)
    vid <- emitOp "stablehlo.convolution"
            [tensorValue input, tensorValue kernel]
            [inType1, inType2]
            [ AttrIntList "input_spatial_dimensions" [1, 2]
            , AttrIntList "kernel_spatial_dimensions" [0, 1]
            , AttrIntList "output_spatial_dimensions" [1, 2]
            , AttrInt "input_batch_dimension" 0
            , AttrInt "input_feature_dimension" 3
            , AttrInt "kernel_input_feature_dimension" 2
            , AttrInt "kernel_output_feature_dimension" 3
            , AttrInt "output_batch_dimension" 0
            , AttrInt "output_feature_dimension" 3
            , AttrIntList "window_strides" strideVals
            , AttrIntList "padding" padVals
            , AttrIntList "lhs_dilation" [1, 1]
            , AttrIntList "rhs_dilation" [1, 1]
            , AttrInt "batch_group_count" 1
            , AttrInt "feature_group_count" 1
            ] outType
    return (Tensor vid)

-- | Batch normalization for inference.
--
-- * @x@ has shape @[N, H, W, C]@ (NHWC)
-- * @scale@, @offset@, @mean@, @variance@ each have shape @[C]@
-- * Result has the same shape as @x@
batchNormInference :: forall n h w c.
                      (KnownNat n, KnownNat h, KnownNat w, KnownNat c)
                   => Tensor '[n, h, w, c] 'F32
                   -> Tensor '[c] 'F32   -- scale
                   -> Tensor '[c] 'F32   -- offset
                   -> Tensor '[c] 'F32   -- mean
                   -> Tensor '[c] 'F32   -- variance
                   -> Builder (Tensor '[n, h, w, c] 'F32)
batchNormInference x scale offset mean variance = do
    let chType = tensorType (Proxy @'[c]) (Proxy @'F32)

    -- epsilon as scalar constant, broadcast to [c]
    epsScalar <- constant @'[] @'F32 1.0e-5
    epsCh     <- broadcastWithDims @'[] @'[c] [] epsScalar

    -- varPlusEps = variance + epsilon
    varPlusEps <- add variance epsCh

    -- sqrtVar = sqrt(varPlusEps)
    let (Tensor varPlusEpsVid) = varPlusEps
    sqrtVarVid <- emitOp "stablehlo.sqrt" [varPlusEpsVid] [chType] [] chType
    let sqrtVar = Tensor sqrtVarVid :: Tensor '[c] 'F32

    -- Broadcast [c] params to [n,h,w,c] via dims=[3] (feature dim)
    meanB    <- broadcastWithDims @'[c] @'[n, h, w, c] [3] mean
    sqrtVarB <- broadcastWithDims @'[c] @'[n, h, w, c] [3] sqrtVar
    scaleB   <- broadcastWithDims @'[c] @'[n, h, w, c] [3] scale
    offsetB  <- broadcastWithDims @'[c] @'[n, h, w, c] [3] offset

    -- y = scale * (x - mean) / sqrt(var + eps) + offset
    xMinusMean <- sub x meanB
    normalized <- divide xMinusMean sqrtVarB
    scaled     <- multiply scaleB normalized
    result     <- add scaled offsetB

    return result

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

returnTuple3 :: Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tuple3 s1 d1 s2 d2 s3 d3)
returnTuple3 t1 t2 t3 = return (Tuple3 t1 t2 t3)

returnTuple4 :: Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4)
returnTuple4 t1 t2 t3 t4 = return (Tuple4 t1 t2 t3 t4)

returnTuple5 :: Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5)
returnTuple5 t1 t2 t3 t4 t5 = return (Tuple5 t1 t2 t3 t4 t5)

returnTuple6 :: Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6)
returnTuple6 t1 t2 t3 t4 t5 t6 = return (Tuple6 t1 t2 t3 t4 t5 t6)

returnTuple7 :: Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7 -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7)
returnTuple7 t1 t2 t3 t4 t5 t6 t7 = return (Tuple7 t1 t2 t3 t4 t5 t6 t7)

returnTuple8 :: Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7 -> Tensor s8 d8 -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8)
returnTuple8 t1 t2 t3 t4 t5 t6 t7 t8 = return (Tuple8 t1 t2 t3 t4 t5 t6 t7 t8)

-- | Return a heterogeneous tuple of tensors from a multi-result builder.
returnT :: Tuple ss ds -> Builder (Tuple ss ds)
returnT = return

-- ---------------------------------------------------------------------------
-- Control flow
-- ---------------------------------------------------------------------------

-- | Single-tensor while loop.
--
-- * @init@ is the initial loop-carried value.
-- * @cond@ takes the loop variable and returns a boolean scalar.
-- * @body@ takes the loop variable and returns the updated value.
-- * Result has the same shape and dtype as @init@.
whileLoop :: forall s d.
             (KnownShape s, KnownDType d)
          => Tensor s d
          -> (Tensor s d -> Builder (Tensor '[] 'Bool))
          -> (Tensor s d -> Builder (Tensor s d))
          -> Builder (Tensor s d)
whileLoop init cond body = do
    let ttype     = tensorType (Proxy @s) (Proxy @d)
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)

    -- Build cond region: ^bb0(%argN: ttype) -> tensor<i1>
    condBlock <- runBlockBuilder [ttype] $ do
        loopVar <- arg @s @d
        condResult <- cond loopVar
        emitReturn [tensorValue condResult] [boolType]

    -- Build body region: ^bb0(%argN: ttype) -> ttype
    bodyBlock <- runBlockBuilder [ttype] $ do
        loopVar <- arg @s @d
        bodyResult <- body loopVar
        emitReturn [tensorValue bodyResult] [ttype]

    let (Tensor initVid) = init
    vid <- emitOpRegions "stablehlo.while" [initVid] [ttype] []
            [Region [condBlock], Region [bodyBlock]] ttype
    return (Tensor vid)

-- | If-then-else selecting between two tensor values.
--
-- Both branches must return a tensor of the same shape and dtype.
conditional :: forall s d.
               (KnownShape s, KnownDType d)
            => Tensor '[] 'Bool
            -> Builder (Tensor s d)   -- true branch thunk
            -> Builder (Tensor s d)   -- false branch thunk
            -> Builder (Tensor s d)
conditional pred trueThunk falseThunk = do
    let ttype    = tensorType (Proxy @s) (Proxy @d)
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)

    -- Build true region (no block args)
    trueBlock <- runBlockBuilder [] $ do
        trueResult <- trueThunk
        emitReturn [tensorValue trueResult] [ttype]

    -- Build false region (no block args)
    falseBlock <- runBlockBuilder [] $ do
        falseResult <- falseThunk
        emitReturn [tensorValue falseResult] [ttype]

    let (Tensor predVid) = pred
    vid <- emitOpRegions "stablehlo.if" [predVid] [boolType] []
            [Region [trueBlock], Region [falseBlock]] ttype
    return (Tensor vid)

-- | Element-wise comparison between two tensors.
--
-- @direction@ must be a valid StableHLO comparison direction:
-- @"EQ"@, @"NE"@, @"GE"@, @"GT"@, @"LE"@, @"LT"@.
compare :: forall s d.
           (KnownShape s, KnownDType d)
        => Tensor s d -> Tensor s d -> Text -> Builder (Tensor s 'Bool)
compare (Tensor x) (Tensor y) direction = do
    let inType  = tensorType (Proxy @s) (Proxy @d)
        outType = tensorType (Proxy @s) (Proxy @'Bool)
    vid <- emitOp "stablehlo.compare" [x, y] [inType, inType]
        [ AttrEnum "comparison_direction" direction
        ] outType
    return (Tensor vid)

lessThan :: forall s d.
            (KnownShape s, KnownDType d)
         => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
lessThan x y = compare x y "LT"

greaterThan :: forall s d.
               (KnownShape s, KnownDType d)
            => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
greaterThan x y = compare x y "GT"

equal :: forall s d.
         (KnownShape s, KnownDType d)
      => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
equal x y = compare x y "EQ"

notEqual :: forall s d.
            (KnownShape s, KnownDType d)
         => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
notEqual x y = compare x y "NE"

lessThanOrEqual :: forall s d.
                   (KnownShape s, KnownDType d)
                => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
lessThanOrEqual x y = compare x y "LE"

greaterThanOrEqual :: forall s d.
                      (KnownShape s, KnownDType d)
                   => Tensor s d -> Tensor s d -> Builder (Tensor s 'Bool)
greaterThanOrEqual x y = compare x y "GE"

-- ---------------------------------------------------------------------------
-- Data movement
-- ---------------------------------------------------------------------------

-- | Helper: format an integer list for MLIR attribute text.
intList :: [Int64] -> Text
intList xs = "[" <> T.intercalate ", " ((T.pack . show) <$> xs) <> "]"

-- | Gather slices from a tensor using index arrays.
--
-- See the StableHLO 'gather' spec for the meaning of each attribute.
gather :: forall sOperand sIndices sResult d.
          ( KnownShape sOperand, KnownShape sIndices
          , KnownShape sResult, KnownDType d )
       => Tensor sOperand d
       -> Tensor sIndices 'I64
       -> [Int64]      -- ^ offset_dims
       -> [Int64]     -- ^ collapsed_slice_dims
       -> [Int64]     -- ^ start_index_map
       -> Int64                               -- ^ index_vector_dim
       -> [Int64]     -- ^ slice_sizes
       -> Builder (Tensor sResult d)
gather operand indices offsetDims collapsedSliceDims startIndexMap indexVectorDim sliceSizes = do
    let operandType = tensorType (Proxy @sOperand) (Proxy @d)
        indicesType = tensorType (Proxy @sIndices)  (Proxy @'I64)
        resultType  = tensorType (Proxy @sResult)   (Proxy @d)

    let dimNumbers = AttrRaw $
            "dimension_numbers = #stablehlo.gather<offset_dims = " <> intList offsetDims
            <> ", collapsed_slice_dims = " <> intList collapsedSliceDims
            <> ", start_index_map = " <> intList startIndexMap
            <> ", index_vector_dim = " <> T.pack (show indexVectorDim)
            <> ">"
        sliceSizesAttr = AttrRaw $
            "slice_sizes = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> sliceSizes) <> ">"
        indicesSorted = AttrBool "indices_are_sorted" False

    let (Tensor operandVid) = operand
        (Tensor indicesVid) = indices

    vid <- emitOp "stablehlo.gather" [operandVid, indicesVid]
            [operandType, indicesType]
            [dimNumbers, sliceSizesAttr, indicesSorted]
            resultType
    return (Tensor vid)

-- | Scatter updates into a tensor at indexed positions.
--
-- The @updateFn@ takes two scalar tensors (current value, update value)
-- and returns the combined scalar.  Common choices:
--
-- * Identity (replace): @\_ upd -> return upd@
-- * Add (accumulate):   @\cur upd -> add cur upd@
scatter :: forall sInput sIndices sUpdates sResult d.
           ( KnownShape sInput, KnownShape sIndices
           , KnownShape sUpdates, KnownShape sResult
           , KnownDType d )
        => Tensor sInput d
        -> Tensor sIndices 'I64
        -> Tensor sUpdates d
        -> (Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[] d))
        -> [Int64]     -- ^ update_window_dims
        -> [Int64]       -- ^ inserted_window_dims
        -> [Int64]     -- ^ scatter_dims_to_operand_dims
        -> Int64                               -- ^ index_vector_dim
        -> Builder (Tensor sResult d)
scatter input indices updates updateFn updateWindowDims insertedWindowDims scatterDimsToOperandDims indexVectorDim = do
    let inputType   = tensorType (Proxy @sInput)   (Proxy @d)
        indicesType = tensorType (Proxy @sIndices)  (Proxy @'I64)
        updatesType = tensorType (Proxy @sUpdates)  (Proxy @d)
        resultType  = tensorType (Proxy @sResult)   (Proxy @d)
        elemType    = tensorType (Proxy @'[])       (Proxy @d)

    -- Build update_computation region
    updateBlock <- runBlockBuilder [elemType, elemType] $ do
        cur <- arg @'[] @d
        upd <- arg @'[] @d
        combined <- updateFn cur upd
        emitReturn [tensorValue combined] [elemType]

    let dimNumbers = AttrRaw $
            "scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = " <> intList updateWindowDims
            <> ", inserted_window_dims = " <> intList insertedWindowDims
            <> ", scatter_dims_to_operand_dims = " <> intList scatterDimsToOperandDims
            <> ", index_vector_dim = " <> T.pack (show indexVectorDim)
            <> ">"
        indicesSorted = AttrBool "indices_are_sorted" False
        uniqueIndices = AttrBool "unique_indices" False

    let (Tensor inputVid)   = input
        (Tensor indicesVid) = indices
        (Tensor updatesVid) = updates

    vid <- emitOpRegions "stablehlo.scatter"
            [inputVid, indicesVid, updatesVid]
            [inputType, indicesType, updatesType]
            [dimNumbers, indicesSorted, uniqueIndices]
            [Region [updateBlock]]
            resultType
    return (Tensor vid)

-- | Extract a sub-array from a tensor using statically-computed indices.
--
-- @start@, @limit@, and @stride@ must have the same length as the rank of
-- the input tensor.
slice :: forall sIn sOut d.
         (KnownShape sIn, KnownShape sOut, KnownDType d)
      => Tensor sIn d
      -> Vector (Length sIn) Int64   -- ^ start_indices
      -> Vector (Length sIn) Int64   -- ^ limit_indices
      -> Vector (Length sIn) Int64   -- ^ strides
      -> Builder (Tensor sOut d)
slice operand start limit stride = do
    let inType   = tensorType (Proxy @sIn)  (Proxy @d)
        outType  = tensorType (Proxy @sOut) (Proxy @d)
        startAttr = AttrIntList "start_indices" (VS.toList start)
        limitAttr = AttrIntList "limit_indices" (VS.toList limit)
        strideAttr = AttrIntList "strides" (VS.toList stride)

    let (Tensor operandVid) = operand
    vid <- emitOp "stablehlo.slice" [operandVid] [inType]
            [startAttr, limitAttr, strideAttr] outType
    return (Tensor vid)

-- | Pad a tensor with a padding value.
--
-- @low@, @high@, and @interior@ must have the same length as the rank of
-- the input tensor.
pad :: forall sIn sOut d.
       (KnownShape sIn, KnownShape sOut, KnownDType d)
    => Tensor sIn d
    -> Tensor '[] d   -- ^ padding value (scalar)
    -> Vector (Length sIn) Int64   -- ^ edge_padding_low
    -> Vector (Length sIn) Int64   -- ^ edge_padding_high
    -> Vector (Length sIn) Int64   -- ^ interior_padding
    -> Builder (Tensor sOut d)
pad operand paddingValue low high interior = do
    let inType   = tensorType (Proxy @sIn)  (Proxy @d)
        padType  = tensorType (Proxy @'[])  (Proxy @d)
        outType  = tensorType (Proxy @sOut) (Proxy @d)
        lowAttr  = AttrIntList "edge_padding_low" (VS.toList low)
        highAttr = AttrIntList "edge_padding_high" (VS.toList high)
        intAttr  = AttrIntList "interior_padding" (VS.toList interior)

    let (Tensor operandVid) = operand
        (Tensor padVid)     = paddingValue
    vid <- emitOp "stablehlo.pad" [operandVid, padVid] [inType, padType]
            [lowAttr, highAttr, intAttr] outType
    return (Tensor vid)

-- | Extract a slice from a tensor using dynamically-computed start indices.
dynamicSlice :: forall sIn sOut d.
                (KnownShape sIn, KnownShape sOut, KnownDType d)
             => Tensor sIn d
             -> [Tensor '[] 'I64]   -- ^ start indices (one scalar i64 per dimension)
             -> Vector (Length sOut) Int64   -- ^ slice_sizes
             -> Builder (Tensor sOut d)
dynamicSlice operand startIndices sliceSizes = do
    let inType   = tensorType (Proxy @sIn)  (Proxy @d)
        outType  = tensorType (Proxy @sOut) (Proxy @d)
        sizesAttr = AttrIntList "slice_sizes" (VS.toList sliceSizes)

    let (Tensor operandVid) = operand
        startVids = tensorValue <$> startIndices
        startTypes = replicate (length startIndices) (tensorType (Proxy @'[]) (Proxy @'I64))

    vid <- emitOp "stablehlo.dynamic_slice" (operandVid : startVids) (inType : startTypes)
            [sizesAttr] outType
    return (Tensor vid)

-- | Sort a tensor along a given dimension.
--
-- The @comparator@ takes two scalar elements and returns a boolean scalar
-- (true if the first should come before the second).
sort :: forall s d.
        (KnownShape s, KnownDType d)
     => Tensor s d
     -> Int64   -- ^ dimension to sort along
     -> Bool    -- ^ is_stable
     -> (Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[] 'Bool))
     -> Builder (Tensor s d)
sort operand dimension isStable comparator = do
    let inType  = tensorType (Proxy @s) (Proxy @d)
        elemType = tensorType (Proxy @'[]) (Proxy @d)
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)

    -- Build comparator region
    compBlock <- runBlockBuilder [elemType, elemType] $ do
        a <- arg @'[] @d
        b <- arg @'[] @d
        result <- comparator a b
        emitReturn [tensorValue result] [boolType]

    let dimAttr    = AttrInt "dimension" (fromIntegral dimension)
        stableAttr = AttrBool "is_stable" isStable

    let (Tensor operandVid) = operand
    vid <- emitOpRegions "stablehlo.sort" [operandVid] [inType]
            [dimAttr, stableAttr]
            [Region [compBlock]]
            inType
    return (Tensor vid)

-- | Convert a tensor from one element type to another.
convert :: forall s dIn dOut.
           (KnownShape s, KnownDType dIn, KnownDType dOut)
        => Tensor s dIn -> Builder (Tensor s dOut)
convert (Tensor x) = do
    let inType  = tensorType (Proxy @s) (Proxy @dIn)
        outType = tensorType (Proxy @s) (Proxy @dOut)
    vid <- emitOp "stablehlo.convert" [x] [inType] [] outType
    return (Tensor vid)

-- | Element-wise selection between two tensors based on a boolean predicate.
--
-- All three tensors must have the same shape.
select :: forall s d.
          (KnownShape s, KnownDType d)
       => Tensor s 'Bool   -- ^ predicate
       -> Tensor s d       -- ^ value if true
       -> Tensor s d       -- ^ value if false
       -> Builder (Tensor s d)
select pred onTrue onFalse = do
    let predType = tensorType (Proxy @s) (Proxy @'Bool)
        valType  = tensorType (Proxy @s) (Proxy @d)

    let (Tensor predVid)   = pred
        (Tensor trueVid)   = onTrue
        (Tensor falseVid)  = onFalse

    vid <- emitOp "stablehlo.select"
            [predVid, trueVid, falseVid]
            [predType, valType, valType]
            [] valType
    return (Tensor vid)

-- | Element-wise logical AND.
--
-- Both inputs must be boolean tensors of the same shape.
logicalAnd :: forall s. KnownShape s => Tensor s 'Bool -> Tensor s 'Bool -> Builder (Tensor s 'Bool)
logicalAnd a b = do
    falseVal <- constant @s @'Bool 0.0
    select a b falseVal

-- | Element-wise logical OR.
logicalOr :: forall s. KnownShape s => Tensor s 'Bool -> Tensor s 'Bool -> Builder (Tensor s 'Bool)
logicalOr a b = do
    trueVal <- constant @s @'Bool 1.0
    select a trueVal b

-- | Element-wise logical NOT.
logicalNot :: forall s. KnownShape s => Tensor s 'Bool -> Builder (Tensor s 'Bool)
logicalNot a = do
    falseVal <- constant @s @'Bool 0.0
    trueVal  <- constant @s @'Bool 1.0
    select a falseVal trueVal

-- | Element-wise map over specified dimensions of one or more tensors.
--
-- The @computation@ receives one scalar tensor per input and must produce a
-- scalar tensor of the output type.
map :: forall s dIn dOut.
       (KnownShape s, KnownDType dIn, KnownDType dOut)
    => [Tensor s dIn]               -- ^ input tensors (all same shape)
    -> [Int64]                      -- ^ dimensions to map over
    -> ([Tensor '[] dIn] -> Builder (Tensor '[] dOut))
    -> Builder (Tensor s dOut)
map inputs dimensions computation = do
    let inType  = tensorType (Proxy @s) (Proxy @dIn)
        outType = tensorType (Proxy @s) (Proxy @dOut)
        elemInType  = tensorType (Proxy @'[]) (Proxy @dIn)
        elemOutType = tensorType (Proxy @'[]) (Proxy @dOut)

    -- Build computation region
    compBlock <- runBlockBuilder (replicate (length inputs) elemInType) $ do
        args <- mapM (const (arg @'[] @dIn)) inputs
        result <- computation args
        emitReturn [tensorValue result] [elemOutType]

    let dimsAttr = AttrIntList "dimensions" dimensions

    let inputVids = tensorValue <$> inputs
        inputTypes = replicate (length inputs) inType

    vid <- emitOpRegions "stablehlo.map"
            inputVids
            inputTypes
            [dimsAttr]
            [Region [compBlock]]
            outType
    return (Tensor vid)

-- | Concatenate a list of tensors along a given dimension.
--
-- All inputs must have the same rank and identical dimensions except along
-- the concat axis.
concatenate :: forall sIn sOut d.
               (KnownShape sIn, KnownShape sOut, KnownDType d)
            => Int64            -- ^ dimension to concatenate along
            -> [Tensor sIn d]   -- ^ input tensors
            -> Builder (Tensor sOut d)
concatenate dim inputs = do
    let inType  = tensorType (Proxy @sIn)  (Proxy @d)
        outType = tensorType (Proxy @sOut) (Proxy @d)
        dimAttr = AttrInt "dimension" (fromIntegral dim)

    let inputVids   = tensorValue <$> inputs
        inputTypes  = replicate (length inputs) inType

    vid <- emitOp "stablehlo.concatenate" inputVids inputTypes [dimAttr] outType
    return (Tensor vid)

-- | Concatenate two tensors along a given dimension.
--
-- The two inputs may have different shapes (differing only along the concat
-- axis), unlike 'concatenate' which requires all inputs to share the same
-- type-level shape.
concatenate2 :: forall s1 s2 sOut d.
                (KnownShape s1, KnownShape s2, KnownShape sOut, KnownDType d)
             => Int64 -> Tensor s1 d -> Tensor s2 d -> Builder (Tensor sOut d)
concatenate2 dim a b = do
    let inType1 = tensorType (Proxy @s1) (Proxy @d)
        inType2 = tensorType (Proxy @s2) (Proxy @d)
        outType = tensorType (Proxy @sOut) (Proxy @d)
        dimAttr = AttrInt "dimension" (fromIntegral dim)

    vid <- emitOp "stablehlo.concatenate"
            [tensorValue a, tensorValue b]
            [inType1, inType2]
            [dimAttr] outType
    return (Tensor vid)

-- | Generate a sequence of numbers along the given dimension.
--
-- @iota dim@ produces a tensor where each element along @dim@ is its index
-- in that dimension. All other dimensions have repeated values.
iota :: forall s. (KnownShape s) => Int64 -> Builder (Tensor s 'I64)
iota iotaDim = do
    let outType = tensorType (Proxy @s) (Proxy @'I64)
        dimAttr = AttrInt "iota_dimension" (fromIntegral iotaDim)
    vid <- emitOp "stablehlo.iota" [] [] [dimAttr] outType
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Layer normalization (composite)
-- ---------------------------------------------------------------------------

-- | Layer normalization over the last dimension.
--
-- Input shape @[..., C]@; gamma and beta have shape @[C]@.
-- Normalizes over the last axis (feature dimension).
layerNorm :: forall batch seq dModel.
             (KnownNat batch, KnownNat seq, KnownNat dModel)
          => Tensor '[batch, seq, dModel] 'F32
          -> Tensor '[dModel] 'F32   -- ^ gamma (scale)
          -> Tensor '[dModel] 'F32   -- ^ beta  (shift)
          -> Builder (Tensor '[batch, seq, dModel] 'F32)
layerNorm x gamma beta = do
    let dModelVal = fromIntegral (natVal (Proxy @dModel)) :: Double

    -- mean over last dim: [batch, seq, dModel] -> [batch, seq]
    mean <- reduceSumDim @'[batch, seq, dModel] @'[batch, seq] [2] x
    dModelConst <- constant @'[] @'F32 dModelVal
    dModelBC    <- broadcastWithDims @'[] @'[batch, seq] [] dModelConst
    mean <- divide mean dModelBC

    -- broadcast mean to [batch, seq, dModel]
    meanBC <- broadcastWithDims @'[batch, seq] @'[batch, seq, dModel] [0, 1] mean

    -- x - mean
    centered <- sub x meanBC

    -- var = mean((x - mean)^2)
    sq <- multiply centered centered
    var <- reduceSumDim @'[batch, seq, dModel] @'[batch, seq] [2] sq
    var <- divide var dModelBC

    -- broadcast var
    varBC <- broadcastWithDims @'[batch, seq] @'[batch, seq, dModel] [0, 1] var

    -- rsqrt(var + epsilon) where epsilon = 1e-5
    eps  <- constant @'[] @'F32 1.0e-5
    epsBC <- broadcastWithDims @'[] @'[batch, seq, dModel] [] eps
    varEps <- add varBC epsBC

    -- sqrt(v) = exp(0.5 * log(v))
    logVar <- logarithm varEps
    half   <- constant @'[] @'F32 0.5
    halfBC <- broadcastWithDims @'[] @'[batch, seq, dModel] [] half
    halfLog <- multiply halfBC logVar
    std    <- exponential halfLog

    one <- constant @'[] @'F32 1.0
    oneBC <- broadcastWithDims @'[] @'[batch, seq, dModel] [] one
    invStd <- divide oneBC std

    -- normalize
    normalized <- multiply centered invStd

    -- gamma * normalized + beta
    gammaBC <- broadcastWithDims @'[dModel] @'[batch, seq, dModel] [2] gamma
    betaBC  <- broadcastWithDims @'[dModel] @'[batch, seq, dModel] [2] beta
    scaled  <- multiply normalized gammaBC
    add scaled betaBC

-- ---------------------------------------------------------------------------
-- Global average pooling (composite)
-- ---------------------------------------------------------------------------

-- | Global average pool over spatial dimensions (H, W).
--
-- Input: [N, H, W, C] → Output: [N, C]
globalAvgPool :: forall n h w c.
                 (KnownNat n, KnownNat h, KnownNat w, KnownNat c)
              => Tensor '[n, h, w, c] 'F32
              -> Builder (Tensor '[n, c] 'F32)
globalAvgPool x = do
    -- Work around PJRT CPU partial multi-dim reduce bug by doing two single-dim reductions.
    step1 <- reduceSumDim @'[n, h, w, c] @'[n, w, c] [1] x
    summed <- reduceSumDim @'[n, w, c] @'[n, c] [1] step1
    let hw = fromIntegral (natVal (Proxy @h) * natVal (Proxy @w)) :: Double
    hwVal <- constant @'[] @'F32 hw
    hwBC  <- broadcastWithDims @'[] @'[n, c] [] hwVal
    divide summed hwBC

-- ---------------------------------------------------------------------------
-- GELU activation (composite)
-- ---------------------------------------------------------------------------

-- | GELU activation using the tanh approximation.
--
-- @gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))@
gelu :: forall s. (KnownShape s) => Tensor s 'F32 -> Builder (Tensor s 'F32)
gelu x = do
    coeff   <- constant @'[] @'F32 0.044715
    sqrt2pi <- constant @'[] @'F32 0.7978845608
    half    <- constant @'[] @'F32 0.5
    one     <- constant @'[] @'F32 1.0

    -- Broadcast scalars to shape s
    let sType = tensorType (Proxy @s) (Proxy @'F32)
        bc scalar = broadcastWithDims @'[] @s [] scalar

    coeffBC   <- bc coeff
    sqrt2piBC <- bc sqrt2pi
    halfBC    <- bc half
    oneBC     <- bc one

    x2 <- multiply x x
    x3 <- multiply x x2
    cX3 <- multiply coeffBC x3
    inner <- add x cX3
    scaled <- multiply sqrt2piBC inner
    t <- tanh scaled
    onePlusT <- add oneBC t
    halfX <- multiply halfBC x
    multiply halfX onePlusT

-- ---------------------------------------------------------------------------
-- Reduce window / pooling
-- ---------------------------------------------------------------------------

-- | General reduce-window primitive.
--
-- For max-pooling use @reduction = "stablehlo.maximum"@ and an init value
-- of negative infinity. For sum-pooling (avg-pool precursor) use
-- @reduction = "stablehlo.add"@ and an init value of @0@.
reduceWindow :: forall sIn sOut d.
                (KnownShape sIn, KnownShape sOut, KnownDType d)
             => Vector (Length sIn) Int64        -- ^ window_dimensions
             -> Vector (Length sIn) Int64        -- ^ window_strides
             -> Vector (Length sIn) (Int64, Int64) -- ^ padding: (low, high) per dimension
             -> Text           -- ^ reduction op, e.g. "stablehlo.maximum"
             -> Tensor '[] d   -- ^ init value (scalar)
             -> Tensor sIn d   -- ^ input
             -> Builder (Tensor sOut d)
reduceWindow windowDims strides padding reduction initVal input = do
    let inType   = tensorType (Proxy @sIn)  (Proxy @d)
        outType  = tensorType (Proxy @sOut) (Proxy @d)
        elemType = tensorType (Proxy @'[])  (Proxy @d)

    -- Build reduction region (takes 2 scalars, returns 1 scalar)
    redBlock <- runBlockBuilder [elemType, elemType] $ do
        a <- arg @'[] @d
        b <- arg @'[] @d
        -- We need to dispatch the reduction op dynamically.
        -- For "stablehlo.maximum" and "stablehlo.add" we have direct wrappers.
        result <- case reduction of
            "stablehlo.maximum" -> maximum a b
            "stablehlo.add"     -> add a b
            _ -> error $ "reduceWindow: unsupported reduction: " ++ show reduction
        emitReturn [tensorValue result] [elemType]

    let windowAttr = AttrIntList "window_dimensions" (VS.toList windowDims)
        strideAttr = AttrIntList "window_strides" (VS.toList strides)
        padStr = "[" <> T.intercalate ", " (padPairV <$> VS.toList padding) <> "]"
        paddingAttr = AttrRaw $ "padding = dense<" <> padStr <> "> : tensor<"
            <> T.pack (show (length (shapeVal (Proxy @sIn)))) <> "x2xi64>"

    let (Tensor initVid) = initVal
        (Tensor inputVid) = input

    vid <- emitOpRegions "stablehlo.reduce_window"
            [inputVid, initVid]
            [inType, elemType]
            [windowAttr, strideAttr, paddingAttr]
            [Region [redBlock]]
            outType
    return (Tensor vid)
  where
    padPairV (l, h) = "[" <> T.pack (show l) <> ", " <> T.pack (show h) <> "]"

-- | 2-D max pooling (NHWC).
maxPool :: forall n h w c oh ow.
           (KnownNat n, KnownNat h, KnownNat w, KnownNat c, KnownNat oh, KnownNat ow)
        => V2 Int64   -- ^ kernel [kh, kw]
        -> V2 Int64   -- ^ stride [sh, sw]
        -> P2         -- ^ padding per spatial dim [[pt, pb], [pl, pr]]
        -> Tensor '[n, h, w, c] 'F32
        -> Builder (Tensor '[n, oh, ow, c] 'F32)
maxPool kernel stride padding x = do
    let windowDims = v4 1 (kernel `VS.index` 0) (kernel `VS.index` 1) 1
        strides    = v4 1 (stride `VS.index` 0) (stride `VS.index` 1) 1
        fullPadding = v4 (0, 0)
                         (fst (padding `VS.index` 0), snd (padding `VS.index` 0))
                         (fst (padding `VS.index` 1), snd (padding `VS.index` 1))
                         (0, 0)
    -- init value for max: a very negative number
    initVal <- constant @'[] @'F32 (-1.0e30)
    reduceWindow windowDims strides fullPadding "stablehlo.maximum" initVal x

-- | 2-D average pooling (NHWC), VALID padding only.
--
-- Computes the mean over each pooling window. The output size is determined
-- by the input size, kernel, and stride (no padding).
avgPool :: forall n h w c oh ow.
           (KnownNat n, KnownNat h, KnownNat w, KnownNat c, KnownNat oh, KnownNat ow)
        => V2 Int64   -- ^ kernel [kh, kw]
        -> V2 Int64   -- ^ stride [sh, sw]
        -> Tensor '[n, h, w, c] 'F32
        -> Builder (Tensor '[n, oh, ow, c] 'F32)
avgPool kernel stride x = do
    let windowDims = v4 1 (kernel `VS.index` 0) (kernel `VS.index` 1) 1
        strides    = v4 1 (stride `VS.index` 0) (stride `VS.index` 1) 1
        fullPadding = v4 (0, 0) (0, 0) (0, 0) (0, 0)
        windowSize = fromIntegral (product (VS.toList kernel)) :: Double
    initVal <- constant @'[] @'F32 0.0
    summed <- reduceWindow windowDims strides fullPadding "stablehlo.add" initVal x
    divisor <- constant @'[] @'F32 windowSize
    divisorBC <- broadcastWithDims @'[] @'[n, oh, ow, c] [] divisor
    divide summed divisorBC

-- ---------------------------------------------------------------------------
-- Transposed convolution (upsampling)
-- ---------------------------------------------------------------------------

-- | 2-D transposed convolution (NHWC) for upsampling.
--
-- This is implemented as a regular convolution with @lhs_dilation@ > 1 on
-- the spatial dimensions, which effectively spaces out the input pixels and
-- produces a larger output.
transposeConvolution :: forall batch h w inCh outCh kh kw oh ow.
                        ( KnownNat batch, KnownNat h, KnownNat w
                        , KnownNat inCh, KnownNat outCh
                        , KnownNat kh, KnownNat kw, KnownNat oh, KnownNat ow )
                     => V2 Int64   -- ^ lhs_dilation (upsample factor), e.g. [2,2]
                     -> P2         -- ^ padding per spatial dim, e.g. p2 (1,1) (1,1) for 2x2 kernel
                     -> Tensor '[batch, h, w, inCh] 'F32
                     -> Tensor '[kh, kw, outCh, inCh] 'F32
                     -> Builder (Tensor '[batch, oh, ow, outCh] 'F32)
transposeConvolution lhsDilation padding input kernel = do
    let inType1 = tensorType (Proxy @'[batch, h, w, inCh]) (Proxy @'F32)
        inType2 = tensorType (Proxy @'[kh, kw, outCh, inCh]) (Proxy @'F32)
        outType = tensorType (Proxy @'[batch, oh, ow, outCh]) (Proxy @'F32)
        padVals = concatMap (\(l, h) -> [l, h]) (VS.toList padding)
        dilVals = VS.toList lhsDilation
    vid <- emitOp "stablehlo.convolution" [tensorValue input, tensorValue kernel]
            [inType1, inType2]
            [ AttrIntList "input_spatial_dimensions" [1, 2]
            , AttrIntList "kernel_spatial_dimensions" [0, 1]
            , AttrIntList "output_spatial_dimensions" [1, 2]
            , AttrInt "input_batch_dimension" 0
            , AttrInt "input_feature_dimension" 3
            , AttrInt "kernel_input_feature_dimension" 3
            , AttrInt "kernel_output_feature_dimension" 2
            , AttrInt "output_batch_dimension" 0
            , AttrInt "output_feature_dimension" 3
            , AttrIntList "window_strides" [1, 1]
            , AttrIntList "padding" padVals
            , AttrIntList "lhs_dilation" dilVals
            , AttrIntList "rhs_dilation" [1, 1]
            , AttrInt "batch_group_count" 1
            , AttrInt "feature_group_count" 1
            ] outType
    return (Tensor vid)


-- ---------------------------------------------------------------------------
-- Multi-value control flow
-- ---------------------------------------------------------------------------

-- | While loop carrying two tensors of potentially different shapes/dtypes.
--
-- Example (count up to 10 while accumulating a sum):
-- @
--   result <- whileLoop2 (constant @'[] @'I64 0) (constant @'[] @'I64 0)
--       (\c s -> lessThan c ten)
--       (\c s -> do
--           c' <- add c one
--           s' <- add s c
--           returnTuple2 c' s')
-- @
whileLoop2 :: forall s1 d1 s2 d2.
              (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
           => Tensor s1 d1 -> Tensor s2 d2
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tuple2 s1 d1 s2 d2))
           -> Builder (Tuple2 s1 d1 s2 d2)
whileLoop2 init1 init2 cond body = do
    let initVids  = [tensorValue init1, tensorValue init2]
        initTypes = [ tensorType (Proxy @s1) (Proxy @d1)
                    , tensorType (Proxy @s2) (Proxy @d2)
                    ]
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)

    -- Build cond region
    condBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1
        v2 <- arg @s2 @d2
        c  <- cond v1 v2
        emitReturn [tensorValue c] [boolType]

    -- Build body region
    bodyBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1
        v2 <- arg @s2 @d2
        Tuple2 r1 r2 <- body v1 v2
        emitReturn [tensorValue r1, tensorValue r2] initTypes

    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    case vids of
        [vid1, vid2] -> return (Tuple2 (Tensor vid1) (Tensor vid2))
        _            -> error "whileLoop2: expected exactly two results"

whileLoop3 :: forall s1 d1 s2 d2 s3 d3.
              (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3)
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tuple3 s1 d1 s2 d2 s3 d3))
           -> Builder (Tuple3 s1 d1 s2 d2 s3 d3)
whileLoop3 init1 init2 init3 cond body = do
    let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3]
        initTypes = [ tensorType (Proxy @s1) (Proxy @d1)
                    , tensorType (Proxy @s2) (Proxy @d2)
                    , tensorType (Proxy @s3) (Proxy @d3)
                    ]
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
    condBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1
        v2 <- arg @s2 @d2
        v3 <- arg @s3 @d3
        c  <- cond v1 v2 v3
        emitReturn [tensorValue c] [boolType]
    bodyBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1
        v2 <- arg @s2 @d2
        v3 <- arg @s3 @d3
        Tuple3 r1 r2 r3 <- body v1 v2 v3
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3] initTypes
    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    case vids of
        [vid1, vid2, vid3] -> return (Tuple3 (Tensor vid1) (Tensor vid2) (Tensor vid3))
        _                  -> error "whileLoop3: expected exactly three results"

whileLoop4 :: forall s1 d1 s2 d2 s3 d3 s4 d4.
              (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4)
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4))
           -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4)
whileLoop4 init1 init2 init3 init4 cond body = do
    let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3, tensorValue init4]
        initTypes = [ tensorType (Proxy @s1) (Proxy @d1)
                    , tensorType (Proxy @s2) (Proxy @d2)
                    , tensorType (Proxy @s3) (Proxy @d3)
                    , tensorType (Proxy @s4) (Proxy @d4)
                    ]
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
    condBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4
        c  <- cond v1 v2 v3 v4
        emitReturn [tensorValue c] [boolType]
    bodyBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4
        Tuple4 r1 r2 r3 r4 <- body v1 v2 v3 v4
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4] initTypes
    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    case vids of
        [vid1, vid2, vid3, vid4] -> return (Tuple4 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4))
        _                        -> error "whileLoop4: expected exactly four results"

whileLoop5 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5.
              (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5)
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5))
           -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5)
whileLoop5 init1 init2 init3 init4 init5 cond body = do
    let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3, tensorValue init4, tensorValue init5]
        initTypes = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5) ]
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
    condBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5
        c  <- cond v1 v2 v3 v4 v5
        emitReturn [tensorValue c] [boolType]
    bodyBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5
        Tuple5 r1 r2 r3 r4 r5 <- body v1 v2 v3 v4 v5
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5] initTypes
    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    case vids of
        [vid1, vid2, vid3, vid4, vid5] -> return (Tuple5 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5))
        _                              -> error "whileLoop5: expected exactly five results"

whileLoop6 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6.
              (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6)
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6))
           -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6)
whileLoop6 init1 init2 init3 init4 init5 init6 cond body = do
    let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3, tensorValue init4, tensorValue init5, tensorValue init6]
        initTypes = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5), tensorType (Proxy @s6) (Proxy @d6) ]
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
    condBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5; v6 <- arg @s6 @d6
        c  <- cond v1 v2 v3 v4 v5 v6
        emitReturn [tensorValue c] [boolType]
    bodyBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5; v6 <- arg @s6 @d6
        Tuple6 r1 r2 r3 r4 r5 r6 <- body v1 v2 v3 v4 v5 v6
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6] initTypes
    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    case vids of
        [vid1, vid2, vid3, vid4, vid5, vid6] -> return (Tuple6 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5) (Tensor vid6))
        _                                    -> error "whileLoop6: expected exactly six results"

whileLoop7 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7.
              (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7)
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7 -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7))
           -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7)
whileLoop7 init1 init2 init3 init4 init5 init6 init7 cond body = do
    let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3, tensorValue init4, tensorValue init5, tensorValue init6, tensorValue init7]
        initTypes = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5), tensorType (Proxy @s6) (Proxy @d6), tensorType (Proxy @s7) (Proxy @d7) ]
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
    condBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5; v6 <- arg @s6 @d6; v7 <- arg @s7 @d7
        c  <- cond v1 v2 v3 v4 v5 v6 v7
        emitReturn [tensorValue c] [boolType]
    bodyBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5; v6 <- arg @s6 @d6; v7 <- arg @s7 @d7
        Tuple7 r1 r2 r3 r4 r5 r6 r7 <- body v1 v2 v3 v4 v5 v6 v7
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6, tensorValue r7] initTypes
    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    case vids of
        [vid1, vid2, vid3, vid4, vid5, vid6, vid7] -> return (Tuple7 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5) (Tensor vid6) (Tensor vid7))
        _                                          -> error "whileLoop7: expected exactly seven results"

whileLoop8 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8.
              (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7, KnownShape s8, KnownDType d8)
           => Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7 -> Tensor s8 d8
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7 -> Tensor s8 d8 -> Builder (Tensor '[] 'Bool))
           -> (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Tensor s4 d4 -> Tensor s5 d5 -> Tensor s6 d6 -> Tensor s7 d7 -> Tensor s8 d8 -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8))
           -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8)
whileLoop8 init1 init2 init3 init4 init5 init6 init7 init8 cond body = do
    let initVids  = [tensorValue init1, tensorValue init2, tensorValue init3, tensorValue init4, tensorValue init5, tensorValue init6, tensorValue init7, tensorValue init8]
        initTypes = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5), tensorType (Proxy @s6) (Proxy @d6), tensorType (Proxy @s7) (Proxy @d7), tensorType (Proxy @s8) (Proxy @d8) ]
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
    condBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5; v6 <- arg @s6 @d6; v7 <- arg @s7 @d7; v8 <- arg @s8 @d8
        c  <- cond v1 v2 v3 v4 v5 v6 v7 v8
        emitReturn [tensorValue c] [boolType]
    bodyBlock <- runBlockBuilder initTypes $ do
        v1 <- arg @s1 @d1; v2 <- arg @s2 @d2; v3 <- arg @s3 @d3; v4 <- arg @s4 @d4; v5 <- arg @s5 @d5; v6 <- arg @s6 @d6; v7 <- arg @s7 @d7; v8 <- arg @s8 @d8
        Tuple8 r1 r2 r3 r4 r5 r6 r7 r8 <- body v1 v2 v3 v4 v5 v6 v7 v8
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6, tensorValue r7, tensorValue r8] initTypes
    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    case vids of
        [vid1, vid2, vid3, vid4, vid5, vid6, vid7, vid8] -> return (Tuple8 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5) (Tensor vid6) (Tensor vid7) (Tensor vid8))
        _                                                -> error "whileLoop8: expected exactly eight results"

-- | While loop carrying N tensors of the same shape and dtype.
--
-- This is useful for batch loops or when all carried values are homogeneous.
whileLoopN :: forall s d.
              (KnownShape s, KnownDType d)
           => [Tensor s d]                          -- ^ initial values
           -> ([Tensor s d] -> Builder (Tensor '[] 'Bool))  -- ^ condition
           -> ([Tensor s d] -> Builder [Tensor s d])        -- ^ body
           -> Builder [Tensor s d]
whileLoopN inits cond body = do
    let ttype     = tensorType (Proxy @s) (Proxy @d)
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
        numVals   = length inits
        initVids  = tensorValue <$> inits
        initTypes = replicate numVals ttype

    -- Build cond region
    condBlock <- runBlockBuilder initTypes $ do
        args <- mapM (\_ -> arg @s @d) [1..numVals]
        c <- cond args
        emitReturn [tensorValue c] [boolType]

    -- Build body region
    bodyBlock <- runBlockBuilder initTypes $ do
        args <- mapM (\_ -> arg @s @d) [1..numVals]
        results <- body args
        emitReturn (tensorValue <$> results) initTypes

    vids <- emitOpRegionsN "stablehlo.while" initVids initTypes []
              [Region [condBlock], Region [bodyBlock]] initTypes
    return (Tensor <$> vids)

-- | If-then-else selecting between two pairs of tensor values.
conditional2 :: forall s1 d1 s2 d2.
                (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
             => Tensor '[] 'Bool
             -> Builder (Tuple2 s1 d1 s2 d2)   -- ^ true branch
             -> Builder (Tuple2 s1 d1 s2 d2)   -- ^ false branch
             -> Builder (Tuple2 s1 d1 s2 d2)
conditional2 pred trueThunk falseThunk = do
    let ttype1  = tensorType (Proxy @s1) (Proxy @d1)
        ttype2  = tensorType (Proxy @s2) (Proxy @d2)
        types   = [ttype1, ttype2]
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)

    -- Build true region (no block args)
    trueBlock <- runBlockBuilder [] $ do
        Tuple2 r1 r2 <- trueThunk
        emitReturn [tensorValue r1, tensorValue r2] types

    -- Build false region (no block args)
    falseBlock <- runBlockBuilder [] $ do
        Tuple2 r1 r2 <- falseThunk
        emitReturn [tensorValue r1, tensorValue r2] types

    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    case vids of
        [vid1, vid2] -> return (Tuple2 (Tensor vid1) (Tensor vid2))
        _            -> error "conditional2: expected exactly two results"

conditional3 :: forall s1 d1 s2 d2 s3 d3.
                (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3)
             => Tensor '[] 'Bool
             -> Builder (Tuple3 s1 d1 s2 d2 s3 d3)
             -> Builder (Tuple3 s1 d1 s2 d2 s3 d3)
             -> Builder (Tuple3 s1 d1 s2 d2 s3 d3)
conditional3 pred trueThunk falseThunk = do
    let ttype1 = tensorType (Proxy @s1) (Proxy @d1)
        ttype2 = tensorType (Proxy @s2) (Proxy @d2)
        ttype3 = tensorType (Proxy @s3) (Proxy @d3)
        types  = [ttype1, ttype2, ttype3]
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)
    trueBlock <- runBlockBuilder [] $ do
        Tuple3 r1 r2 r3 <- trueThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3] types
    falseBlock <- runBlockBuilder [] $ do
        Tuple3 r1 r2 r3 <- falseThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3] types
    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    case vids of
        [vid1, vid2, vid3] -> return (Tuple3 (Tensor vid1) (Tensor vid2) (Tensor vid3))
        _                  -> error "conditional3: expected exactly three results"

conditional4 :: forall s1 d1 s2 d2 s3 d3 s4 d4.
                (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4)
             => Tensor '[] 'Bool
             -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4)
             -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4)
             -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4)
conditional4 pred trueThunk falseThunk = do
    let ttype1 = tensorType (Proxy @s1) (Proxy @d1)
        ttype2 = tensorType (Proxy @s2) (Proxy @d2)
        ttype3 = tensorType (Proxy @s3) (Proxy @d3)
        ttype4 = tensorType (Proxy @s4) (Proxy @d4)
        types  = [ttype1, ttype2, ttype3, ttype4]
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)
    trueBlock <- runBlockBuilder [] $ do
        Tuple4 r1 r2 r3 r4 <- trueThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4] types
    falseBlock <- runBlockBuilder [] $ do
        Tuple4 r1 r2 r3 r4 <- falseThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4] types
    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    case vids of
        [vid1, vid2, vid3, vid4] -> return (Tuple4 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4))
        _                        -> error "conditional4: expected exactly four results"

conditional5 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5.
                (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5)
             => Tensor '[] 'Bool
             -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5)
             -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5)
             -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5)
conditional5 pred trueThunk falseThunk = do
    let types = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5) ]
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)
    trueBlock <- runBlockBuilder [] $ do
        Tuple5 r1 r2 r3 r4 r5 <- trueThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5] types
    falseBlock <- runBlockBuilder [] $ do
        Tuple5 r1 r2 r3 r4 r5 <- falseThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5] types
    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    case vids of
        [vid1, vid2, vid3, vid4, vid5] -> return (Tuple5 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5))
        _                              -> error "conditional5: expected exactly five results"

conditional6 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6.
                (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6)
             => Tensor '[] 'Bool
             -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6)
             -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6)
             -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6)
conditional6 pred trueThunk falseThunk = do
    let types = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5), tensorType (Proxy @s6) (Proxy @d6) ]
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)
    trueBlock <- runBlockBuilder [] $ do
        Tuple6 r1 r2 r3 r4 r5 r6 <- trueThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6] types
    falseBlock <- runBlockBuilder [] $ do
        Tuple6 r1 r2 r3 r4 r5 r6 <- falseThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6] types
    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    case vids of
        [vid1, vid2, vid3, vid4, vid5, vid6] -> return (Tuple6 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5) (Tensor vid6))
        _                                    -> error "conditional6: expected exactly six results"

conditional7 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7.
                (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7)
             => Tensor '[] 'Bool
             -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7)
             -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7)
             -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7)
conditional7 pred trueThunk falseThunk = do
    let types = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5), tensorType (Proxy @s6) (Proxy @d6), tensorType (Proxy @s7) (Proxy @d7) ]
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)
    trueBlock <- runBlockBuilder [] $ do
        Tuple7 r1 r2 r3 r4 r5 r6 r7 <- trueThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6, tensorValue r7] types
    falseBlock <- runBlockBuilder [] $ do
        Tuple7 r1 r2 r3 r4 r5 r6 r7 <- falseThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6, tensorValue r7] types
    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    case vids of
        [vid1, vid2, vid3, vid4, vid5, vid6, vid7] -> return (Tuple7 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5) (Tensor vid6) (Tensor vid7))
        _                                          -> error "conditional7: expected exactly seven results"

conditional8 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8.
                (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7, KnownShape s8, KnownDType d8)
             => Tensor '[] 'Bool
             -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8)
             -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8)
             -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8)
conditional8 pred trueThunk falseThunk = do
    let types = [ tensorType (Proxy @s1) (Proxy @d1), tensorType (Proxy @s2) (Proxy @d2), tensorType (Proxy @s3) (Proxy @d3), tensorType (Proxy @s4) (Proxy @d4), tensorType (Proxy @s5) (Proxy @d5), tensorType (Proxy @s6) (Proxy @d6), tensorType (Proxy @s7) (Proxy @d7), tensorType (Proxy @s8) (Proxy @d8) ]
        boolType = tensorType (Proxy @'[]) (Proxy @'Bool)
    trueBlock <- runBlockBuilder [] $ do
        Tuple8 r1 r2 r3 r4 r5 r6 r7 r8 <- trueThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6, tensorValue r7, tensorValue r8] types
    falseBlock <- runBlockBuilder [] $ do
        Tuple8 r1 r2 r3 r4 r5 r6 r7 r8 <- falseThunk
        emitReturn [tensorValue r1, tensorValue r2, tensorValue r3, tensorValue r4, tensorValue r5, tensorValue r6, tensorValue r7, tensorValue r8] types
    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    case vids of
        [vid1, vid2, vid3, vid4, vid5, vid6, vid7, vid8] -> return (Tuple8 (Tensor vid1) (Tensor vid2) (Tensor vid3) (Tensor vid4) (Tensor vid5) (Tensor vid6) (Tensor vid7) (Tensor vid8))
        _                                                -> error "conditional8: expected exactly eight results"

-- | If-then-else selecting between N tensor values of the same shape/dtype.
--
-- The caller must provide the expected number of results so the regions
-- can be built with the correct return arity.
conditionalN :: forall s d.
                (KnownShape s, KnownDType d)
             => Int                    -- ^ number of results (determines branch arity)
             -> Tensor '[] 'Bool
             -> Builder [Tensor s d]   -- ^ true branch
             -> Builder [Tensor s d]   -- ^ false branch
             -> Builder [Tensor s d]
conditionalN n pred trueThunk falseThunk = do
    let ttype     = tensorType (Proxy @s) (Proxy @d)
        boolType  = tensorType (Proxy @'[]) (Proxy @'Bool)
        types     = replicate n ttype

    -- Build true region
    trueBlock <- runBlockBuilder [] $ do
        results <- trueThunk
        emitReturn (tensorValue <$> take n results) types

    -- Build false region
    falseBlock <- runBlockBuilder [] $ do
        results <- falseThunk
        emitReturn (tensorValue <$> take n results) types

    let (Tensor predVid) = pred
    vids <- emitOpRegionsN "stablehlo.if" [predVid] [boolType] []
              [Region [trueBlock], Region [falseBlock]] types
    return (Tensor <$> vids)

-- ---------------------------------------------------------------------------
-- Random number generation
-- ---------------------------------------------------------------------------

-- | Generate a tensor of uniformly distributed random values in @[a, b)@.
--
-- The @shape@ operand is a constant 1-D @i64@ tensor built from the
-- type-level shape.  The output dtype is @F32@.
--
-- Emits @stablehlo.rng@ with @distribution = UNIFORM@.
rngUniform :: forall s. KnownShape s
           => Tensor '[] 'F32   -- ^ lower bound @a@
           -> Tensor '[] 'F32   -- ^ upper bound @b@
           -> Builder (Tensor s 'F32)
rngUniform a b = do
    let outType = tensorType (Proxy @s) (Proxy @'F32)
        shapeVals = fromIntegral <$> shapeVal (Proxy @s) :: [Int64]
    -- Build a constant i64 tensor for the shape operand.
    -- We emit it as a stablehlo.constant then use it as an operand.
    shapeConst <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [fromIntegral (length shapeVals)] I64 (fromIntegral <$> shapeVals)]
        (TensorType [Just (fromIntegral (length shapeVals))] I64)
    vid <- emitOp "stablehlo.rng"
            [tensorValue a, tensorValue b, shapeConst]
            [ TensorType [] F32, TensorType [] F32, TensorType [Just (fromIntegral (length shapeVals))] I64 ]
            [AttrEnum "rng_distribution" "UNIFORM"]
            outType
    return (Tensor vid)

-- | Generate a tensor of standard normal random values (mean 0, std 1).
--
-- Emits @stablehlo.rng@ with @distribution = NORMAL@.
rngNormal :: forall s. KnownShape s
          => Builder (Tensor s 'F32)
rngNormal = do
    let outType = tensorType (Proxy @s) (Proxy @'F32)
        shapeVals = fromIntegral <$> shapeVal (Proxy @s) :: [Int64]
    a <- constant @'[] @'F32 0.0
    b <- constant @'[] @'F32 1.0
    shapeConst <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [fromIntegral (length shapeVals)] I64 (fromIntegral <$> shapeVals)]
        (TensorType [Just (fromIntegral (length shapeVals))] I64)
    vid <- emitOp "stablehlo.rng"
            [tensorValue a, tensorValue b, shapeConst]
            [ TensorType [] F32, TensorType [] F32, TensorType [Just (fromIntegral (length shapeVals))] I64 ]
            [AttrEnum "rng_distribution" "NORMAL"]
            outType
    return (Tensor vid)

-- | Deterministic random bit generator using the Threefry algorithm.
--
-- Takes a 2-element @UI64@ state tensor and returns @(new_state, output)@.
-- The output is filled with random bits of type @UI64@.
--
-- Emits @stablehlo.rng_bit_generator@ with @algorithm = THREE_FRY@.
rngBitGenerator :: forall s. (KnownShape s, KnownDType 'UI64)
                => Tensor '[2] 'UI64   -- ^ initial state
                -> Builder (Tensor '[2] 'UI64, Tensor s 'UI64)
rngBitGenerator state = do
    let stateType = tensorType (Proxy @'[2]) (Proxy @'UI64)
        outType   = tensorType (Proxy @s) (Proxy @'UI64)
    vids <- emitOpN "stablehlo.rng_bit_generator"
              [tensorValue state]
              [stateType]
              [AttrEnum "rng_algorithm" "THREE_FRY"]
              [stateType, outType]
    case vids of
        [vidState, vidOut] -> return (Tensor vidState, Tensor vidOut)
        _                  -> error "rngBitGenerator: expected exactly two results"

-- ---------------------------------------------------------------------------
-- Composite / convenience ops
-- ---------------------------------------------------------------------------

-- | Sigmoid activation: @1 / (1 + exp(-x))@.
sigmoid :: forall s. KnownShape s => Tensor s 'F32 -> Builder (Tensor s 'F32)
sigmoid x = do
    negX    <- negate x
    expNegX <- exponential negX
    one     <- constant @s @'F32 1.0
    denom   <- add one expNegX
    divide one denom

-- | Reduce-sum over all dimensions, producing a scalar.
sumAll :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor '[] d)
sumAll = reduceSum

-- | Extract a single scalar element from a 1-D tensor at a constant index.
slice1 :: forall n d. (KnownShape '[n], KnownDType d)
       => Tensor '[n] d -> Int64 -> Builder (Tensor '[] d)
slice1 vec i = do
    sliced <- slice @'[n] @'[1] @d vec (v1 i) (v1 (i + 1)) (v1 1)
    reshape @'[1] @'[] sliced

-- | Pack two scalar tensors into a rank-1 tensor of shape @[2]@.
pack2 :: forall d. KnownDType d
      => Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[2] d)
pack2 x y = do
    x1 <- reshape @'[] @'[1] x
    y1 <- reshape @'[] @'[1] y
    concatenate @'[1] @'[2] @d 0 [x1, y1]

-- | Pack three scalar tensors into a rank-1 tensor of shape @[3]@.
pack3 :: forall d. KnownDType d
      => Tensor '[] d -> Tensor '[] d -> Tensor '[] d -> Builder (Tensor '[3] d)
pack3 x y z = do
    x1 <- reshape @'[] @'[1] x
    y1 <- reshape @'[] @'[1] y
    z1 <- reshape @'[] @'[1] z
    concatenate @'[1] @'[3] @d 0 [x1, y1, z1]

-- ---------------------------------------------------------------------------
-- Product reductions
-- ---------------------------------------------------------------------------

-- | Product of all elements (reduce over all dimensions).
productAll :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> Builder (Tensor '[] d)
productAll (Tensor x) = do
    let dims = [0 .. fromIntegral (length (shapeVal (Proxy @s))) - 1]
    productDim @s @'[] dims (Tensor x)

-- | Product elements over specific dimensions.
productDim :: forall sFrom sTo d.
              (KnownShape sFrom, KnownShape sTo, KnownDType d)
           => [Int] -> Tensor sFrom d -> Builder (Tensor sTo d)
productDim dims (Tensor x) = do
    let inType   = tensorType (Proxy @sFrom) (Proxy @d)
        outType  = tensorType (Proxy @sTo)   (Proxy @d)
        elemType = tensorType (Proxy @'[])   (Proxy @d)
    -- Init value for stablehlo.reduce (must be scalar)
    oneVid <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [] (dtypeVal (Proxy @d)) [1.0]] elemType
    -- Build reduction region: two scalar args, apply stablehlo.multiply
    redBlock <- runBlockBuilder [elemType, elemType] $ do
        a <- arg @'[] @d
        b <- arg @'[] @d
        prodVid <- emitOp "stablehlo.multiply"
                    [tensorValue a, tensorValue b]
                    [elemType, elemType] [] elemType
        emitReturn [prodVid] [elemType]
    vid <- emitOpRegions "stablehlo.reduce"
            [x, oneVid]
            [inType, elemType]
            [AttrIntList "dimensions" (fromIntegral <$> dims)]
            [Region [redBlock]]
            outType
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Split and stack
-- ---------------------------------------------------------------------------

-- | Split a tensor into @n@ equal parts along dimension @dim@.
-- The size of @dim@ in the input must be evenly divisible by @n@.
split :: forall sIn sOut d.
         (KnownShape sIn, KnownShape sOut, KnownDType d, KnownNat (Length sIn))
      => Int64            -- ^ dimension to split along
      -> Int64            -- ^ number of equal splits
      -> Tensor sIn d
      -> Builder [Tensor sOut d]
split dim n t = do
    let sInShape = shapeVal (Proxy @sIn)
        rank = length sInShape
        dimSize = fromIntegral (sInShape !! fromIntegral dim) :: Int
        chunkSize = dimSize `div` fromIntegral n
        stride = VS.replicate 1
    when (dimSize `mod` fromIntegral n /= 0) $
        error "split: dimension size not evenly divisible by number of splits"
    Prelude.mapM (\i -> do
        let startList = [if j == fromIntegral dim then fromIntegral (i * chunkSize) else 0 | j <- [0..rank-1]]
            limitList = [if j == fromIntegral dim then fromIntegral ((i+1) * chunkSize) else fromIntegral (sInShape !! j) | j <- [0..rank-1]]
            start = fromJust (VS.fromList startList)
            limit = fromJust (VS.fromList limitList)
        slice @sIn @sOut @d t start limit stride
        ) [0 .. fromIntegral n - 1]

-- | Stack a list of tensors along a new axis.
-- All inputs must have the same shape. The new axis is inserted at position
-- @dim@ in the output shape.
stack :: forall sIn sOut d.
         (KnownShape sIn, KnownShape sOut, KnownDType d)
      => Int64            -- ^ axis to insert and stack along
      -> [Tensor sIn d]   -- ^ tensors to stack (must be non-empty)
      -> Builder (Tensor sOut d)
stack dim inputs = do
    let n = length inputs
        inShape  = fmap fromIntegral (shapeVal (Proxy @sIn)) :: [Int64]
        outShape = fmap fromIntegral (shapeVal (Proxy @sOut)) :: [Int64]
        dt = dtypeVal (Proxy @d)
        rank = length inShape
        expectedOutShape = take (fromIntegral dim) inShape ++ [fromIntegral n] ++ drop (fromIntegral dim) inShape
        reshapedShape = take (fromIntegral dim) inShape ++ [1] ++ drop (fromIntegral dim) inShape
        inType = tensorType (Proxy @sIn) (Proxy @d)
        outType = tensorType (Proxy @sOut) (Proxy @d)
        reshapedType = TensorType (fmap (Just . fromIntegral) reshapedShape) dt
    when (n == 0) $ error "stack: empty input list"
    when (outShape /= expectedOutShape) $
        error $ "stack: output shape mismatch. Expected " ++ show expectedOutShape ++ ", got " ++ show outShape
    reshapedVids <- Prelude.mapM (\(Tensor vid) ->
        emitOp "stablehlo.reshape" [vid] [inType] [] reshapedType
        ) inputs
    let dimAttr = AttrInt "dimension" (fromIntegral dim)
    vid <- emitOp "stablehlo.concatenate" reshapedVids (replicate n reshapedType) [dimAttr] outType
    return (Tensor vid)

-- ---------------------------------------------------------------------------
-- Top-K
-- ---------------------------------------------------------------------------

-- | Return the top-K values along a dimension, in descending order.
--
-- Note: A @topKWithIndices@ variant is future work; it requires multi-operand
-- sort support (sorting values and their index tensors together).
topK :: forall s sOut d.
        (KnownShape s, KnownShape sOut, KnownDType d)
     => Int64            -- ^ K
     -> Int64            -- ^ dimension to sort along
     -> Tensor s d
     -> Builder (Tensor sOut d)
topK k dim t = do
    let sShape = fmap fromIntegral (shapeVal (Proxy @s)) :: [Int64]
        outShape = fmap fromIntegral (shapeVal (Proxy @sOut)) :: [Int64]
        rank = length sShape
        expectedOutShape = [if j == fromIntegral dim then fromIntegral k else sShape !! j | j <- [0..rank-1]]
    when (outShape /= expectedOutShape) $
        error $ "topK: output shape mismatch. Expected " ++ show expectedOutShape ++ ", got " ++ show outShape
    -- Sort descending
    sorted <- sort @s @d t dim False $ \a b -> greaterThan a b
    -- Slice the first K elements along dim
    let start = replicate rank 0
        limit = [if j == fromIntegral dim then fromIntegral k else sShape !! j | j <- [0..rank-1]]
        stride = replicate rank 1
    slice @s @sOut @d sorted (vecFromListUnsafe @s start) (vecFromListUnsafe @s limit) (vecFromListUnsafe @s stride)

-- ---------------------------------------------------------------------------
-- Einsum
-- ---------------------------------------------------------------------------

-- | Einstein summation for two tensors.
--
-- Example: @einsum "ijk,jkl->il" a b@ contracts the @j@ and @k@ dimensions,
-- leaving @i@ from the left operand and @l@ from the right.
--
-- The caller must supply the output shape @sOut@ via type application or
-- inference context. Runtime validation ensures the subscript string agrees
-- with the type-level shapes.
einsum :: forall s1 s2 sOut d.
          (KnownShape s1, KnownShape s2, KnownShape sOut, KnownDType d)
       => String              -- ^ subscript string, e.g. "ijk,jkl->il"
       -> Tensor s1 d
       -> Tensor s2 d
       -> Builder (Tensor sOut d)
einsum spec (Tensor x) (Tensor y) = do
    let (left, right, out) = parseEinsum spec
        s1Shape = shapeVal (Proxy @s1)
        s2Shape = shapeVal (Proxy @s2)
        sOutShape = shapeVal (Proxy @sOut)
        dt = dtypeVal (Proxy @d)
        rank1 = length s1Shape
        rank2 = length s2Shape
        rankOut = length sOutShape
    when (length left /= rank1) $ error "einsum: left subscript rank mismatch"
    when (length right /= rank2) $ error "einsum: right subscript rank mismatch"
    when (length out /= rankOut) $ error "einsum: output subscript rank mismatch"
    let batchLabels     = [c | c <- left, c `elem` right, c `elem` out]
        contractLabels  = [c | c <- left, c `elem` right, c `notElem` out]
        leftFree        = [c | c <- left, c `elem` out, c `notElem` right]
        rightFree       = [c | c <- right, c `elem` out, c `notElem` left]
        natural         = batchLabels ++ leftFree ++ rightFree
        lhsBatch        = fmap (fromIntegral . fromJust . (`elemIndex` left)) batchLabels
        rhsBatch        = fmap (fromIntegral . fromJust . (`elemIndex` right)) batchLabels
        lhsContract     = fmap (fromIntegral . fromJust . (`elemIndex` left)) contractLabels
        rhsContract     = fmap (fromIntegral . fromJust . (`elemIndex` right)) contractLabels
        -- Build natural shape from actual dimensions
        lookupDim label
            | label `elem` left  = s1Shape !! fromJust (label `elemIndex` left)
            | otherwise          = s2Shape !! fromJust (label `elemIndex` right)
        naturalShape    = fmap (fromIntegral . lookupDim) natural
        naturalType     = TensorType (fmap (Just . fromIntegral) naturalShape) dt
        inType1         = tensorType (Proxy @s1) (Proxy @d)
        inType2         = tensorType (Proxy @s2) (Proxy @d)
        outType         = tensorType (Proxy @sOut) (Proxy @d)
        batchL    = AttrIntList "lhs_batching_dimensions" lhsBatch
        batchR    = AttrIntList "rhs_batching_dimensions" rhsBatch
        contractL = AttrIntList "lhs_contracting_dimensions" lhsContract
        contractR = AttrIntList "rhs_contracting_dimensions" rhsContract
        -- Validate output shape
        expectedOutShape = fmap (fromIntegral . lookupDim) out
    when (fmap fromIntegral sOutShape /= expectedOutShape) $
        error $ "einsum: output shape mismatch. Expected " ++ show expectedOutShape ++ ", got " ++ show (fmap fromIntegral sOutShape)
    vid <- emitOp "stablehlo.dot_general" [x, y] [inType1, inType2]
            [batchL, batchR, contractL, contractR] naturalType
    if natural == out
    then return (Tensor vid)
    else do
        let perm = fmap (fromIntegral . fromJust . (`elemIndex` natural)) out
            permAttr = AttrIntList "permutation" perm
        vidT <- emitOp "stablehlo.transpose" [vid] [naturalType] [permAttr] outType
        return (Tensor vidT)
  where
    parseEinsum s =
        case break (== '-') s of
            (lhsRhs, '-':'>':outRest) ->
                case break (== ',') lhsRhs of
                    (left, ',':right) -> (left, right, outRest)
                    _ -> error "einsum: expected two operands separated by comma"
            _ -> error "einsum: expected -> in subscript string"


-- ---------------------------------------------------------------------------
-- Custom calls
-- ---------------------------------------------------------------------------

-- | Single-result custom call. All inputs share the same shape / dtype.
--
-- The target name is the C symbol that XLA will resolve via @dlsym@.
-- Use 'customCallRaw' for heterogeneous input types or multiple results.
customCall1 :: forall s d. (KnownShape s, KnownDType d)
            => Text              -- ^ target symbol name
            -> [Tensor s d]      -- ^ inputs
            -> Text              -- ^ backend_config opaque payload
            -> Bool              -- ^ has_side_effect
            -> Builder (Tensor s d)
customCall1 target inputs backendConfig hasSideEffect = do
    let vids    = tensorValue <$> inputs
        inType  = tensorType (Proxy @s) (Proxy @d)
        outType = tensorType (Proxy @s) (Proxy @d)
    vidRes <- emitCustomCall target vids (replicate (length inputs) inType) backendConfig hasSideEffect 3 [outType]
    case vidRes of
        [vid] -> return (Tensor vid)
        _     -> error "customCall1: expected exactly one result"

-- | Two-result custom call.
customCall2 :: forall s1 d1 s2 d2. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
            => Text
            -> [Tensor s1 d1]    -- ^ inputs (uniform type for convenience)
            -> Text              -- ^ backend_config
            -> Bool              -- ^ has_side_effect
            -> Builder (Tensor s1 d1, Tensor s2 d2)
customCall2 target inputs backendConfig hasSideEffect = do
    let vids     = tensorValue <$> inputs
        inType   = tensorType (Proxy @s1) (Proxy @d1)
        outType1 = tensorType (Proxy @s1) (Proxy @d1)
        outType2 = tensorType (Proxy @s2) (Proxy @d2)
    vidsRes <- emitCustomCall target vids (replicate (length inputs) inType) backendConfig hasSideEffect 3 [outType1, outType2]
    case vidsRes of
        [v1, v2] -> return (Tensor v1, Tensor v2)
        _        -> error "customCall2: expected exactly two results"

-- | Low-level custom call for plugin authors.
--
-- Accepts raw 'ValueId's and 'TensorType's so that callers can mix shapes
-- and dtypes freely (e.g. sparse indices as 'I32' and values as 'F32').
-- The caller is responsible for ensuring the C kernel signature matches.
customCallRaw :: Text
              -> [ValueId]         -- ^ operand value ids
              -> [TensorType]      -- ^ operand types
              -> Text              -- ^ backend_config
              -> Bool              -- ^ has_side_effect
              -> Int32             -- ^ api_version
              -> [TensorType]      -- ^ result types
              -> Builder [ValueId]
customCallRaw = emitCustomCall

-- | Single-result custom call with an associated VJP backward target.
--
-- The forward call carries a @hhlo.vjp_target@ attribute that the autograd
-- engine reads during reverse-mode differentiation.  See 'customVJPRaw' for
-- the low-level primitive.
customVJP1 :: forall s d. (KnownShape s, KnownDType d)
           => Text              -- ^ forward target symbol name
           -> Text              -- ^ backward target symbol name
           -> [Tensor s d]      -- ^ inputs
           -> Text              -- ^ backend_config opaque payload
           -> Bool              -- ^ has_side_effect
           -> Builder (Tensor s d)
customVJP1 fwdTarget bwdTarget inputs backendConfig hasSideEffect = do
    let vids    = tensorValue <$> inputs
        inType  = tensorType (Proxy @s) (Proxy @d)
        outType = tensorType (Proxy @s) (Proxy @d)
    vidRes <- customVJPRaw fwdTarget bwdTarget vids (replicate (length inputs) inType) backendConfig hasSideEffect 3 [outType]
    case vidRes of
        [vid] -> return (Tensor vid)
        _     -> error "customVJP1: expected exactly one result"
