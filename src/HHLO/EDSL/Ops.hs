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
    , dotGeneral
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
    -- * Reductions
    , reduceSum
    , reduceSumDim
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
    , layerNorm
    , globalAvgPool
    , gelu
    -- * Control flow
    , whileLoop
    , whileLoopN
    , whileLoop2
    , conditional
    , conditional2
    , compare
    , lessThan
    , greaterThan
    , equal
    , notEqual
    , lessThanOrEqual
    , greaterThanOrEqual
    -- * Data movement
    , gather
    , scatter
    , slice
    , pad
    , dynamicSlice
    , sort
    , convert
    -- * Selection
    , select
    -- * Map
    , map
    -- * Constants
    , constant
    -- * Tuple
    , Tuple2(..)
    , returnTuple2
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
    ) where

import Prelude hiding (subtract, negate, maximum, minimum, abs, compare, map, tanh, sqrt, sin, cos, tan, floor, ceiling)

import Data.Int (Int64)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
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
    vid <- emitOp "stablehlo.dot" [x, y] [inType1, inType2] [] outType
    return (Tensor vid)

-- | General dot product with explicit batch and contracting dimensions.
-- Use this for batched matrix multiplication (rank > 2).
dotGeneral :: forall s1 s2 sOut d.
              (KnownShape s1, KnownShape s2, KnownShape sOut, KnownDType d)
           => [Int64]   -- ^ lhs batch dims
           -> [Int64]   -- ^ rhs batch dims
           -> [Int64]   -- ^ lhs contracting dims
           -> [Int64]   -- ^ rhs contracting dims
           -> Tensor s1 d
           -> Tensor s2 d
           -> Builder (Tensor sOut d)
dotGeneral lhsBatch rhsBatch lhsContract rhsContract (Tensor x) (Tensor y) = do
    let inType1 = tensorType (Proxy @s1) (Proxy @d)
        inType2 = tensorType (Proxy @s2) (Proxy @d)
        outType = tensorType (Proxy @sOut) (Proxy @d)
        batchAttr      = AttrString "batching_dims" ("[" <> T.intercalate ", " (fmap (T.pack . show) lhsBatch) <> "] x [" <> T.intercalate ", " (fmap (T.pack . show) rhsBatch) <> "]")
        contractingAttr = AttrString "contracting_dims" ("[" <> T.intercalate ", " (fmap (T.pack . show) lhsContract) <> "] x [" <> T.intercalate ", " (fmap (T.pack . show) rhsContract) <> "]")
    vid <- emitOp "stablehlo.dot_general" [x, y] [inType1, inType2]
            [ batchAttr
            , contractingAttr
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
        [AttrIntList "dims" (fromIntegral <$> dims)] outType
    return (Tensor vid)

-- | Transpose a tensor by permuting dimensions.
--
-- @perm@ must be a permutation of @[0 .. rank-1]@.
transpose :: forall sIn sOut d. (KnownShape sIn, KnownShape sOut, KnownDType d)
          => [Int64] -> Tensor sIn d -> Builder (Tensor sOut d)
transpose perm (Tensor x) = do
    let inType  = tensorType (Proxy @sIn)  (Proxy @d)
        outType = tensorType (Proxy @sOut) (Proxy @d)
        permAttr = AttrIntList "permutation" perm
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
            [AttrRaw $ "dimensions = array<i64: " <> T.intercalate ", " [T.pack (show d) | d <- dims] <> ">"]
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
conv2d x k = conv2dWithPadding @batch @h @w @inCh @outCh @kh @kw @oh @ow [1, 1] (replicate 2 [0, 0]) x k

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
                  => [Int64]        -- ^ strides [sh, sw]
                  -> [[Int64]]      -- ^ padding [[pt, pb], [pl, pr]]
                  -> Tensor '[batch, h, w, inCh] 'F32
                  -> Tensor '[kh, kw, inCh, outCh] 'F32
                  -> Builder (Tensor '[batch, oh, ow, outCh] 'F32)
conv2dWithPadding strides padding input kernel = do
    let inType1 = tensorType (Proxy @'[batch, h, w, inCh])   (Proxy @'F32)
        inType2 = tensorType (Proxy @'[kh, kw, inCh, outCh]) (Proxy @'F32)
        outType = tensorType (Proxy @'[batch, oh, ow, outCh]) (Proxy @'F32)
        dimNums = "[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]"
        strideStr = "[" <> T.intercalate ", " ((T.pack . show) <$> strides) <> "]"
        padStr = "[" <> T.intercalate ", " (padPair <$> padding) <> "]"
        window  = "{stride = " <> strideStr <> ", pad = " <> padStr <> "}"
    vid <- emitOp "stablehlo.convolution"
            [tensorValue input, tensorValue kernel]
            [inType1, inType2]
            [ AttrString "dim_numbers" dimNums
            , AttrString "window" window
            , AttrInt "batch_group_count" 1
            , AttrInt "feature_group_count" 1
            ] outType
    return (Tensor vid)
  where
    padPair [l, h] = "[" <> T.pack (show l) <> ", " <> T.pack (show h) <> "]"
    padPair _      = error "conv2dWithPadding: padding must be [[low,high], ...]"

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
        [ AttrRaw ("comparison_direction = #stablehlo<comparison_direction " <> direction <> ">")
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
       -> [Int64]   -- ^ offset_dims
       -> [Int64]   -- ^ collapsed_slice_dims
       -> [Int64]   -- ^ start_index_map
       -> Int64     -- ^ index_vector_dim
       -> [Int64]   -- ^ slice_sizes
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
        -> [Int64]   -- ^ update_window_dims
        -> [Int64]   -- ^ inserted_window_dims
        -> [Int64]   -- ^ scatter_dims_to_operand_dims
        -> Int64     -- ^ index_vector_dim
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
      -> [Int64]   -- ^ start_indices
      -> [Int64]   -- ^ limit_indices
      -> [Int64]   -- ^ strides
      -> Builder (Tensor sOut d)
slice operand start limit stride = do
    let inType   = tensorType (Proxy @sIn)  (Proxy @d)
        outType  = tensorType (Proxy @sOut) (Proxy @d)
        startAttr = AttrRaw $ "start_indices = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> start) <> ">"
        limitAttr = AttrRaw $ "limit_indices = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> limit) <> ">"
        strideAttr = AttrRaw $ "strides = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> stride) <> ">"

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
    -> [Int64]        -- ^ edge_padding_low
    -> [Int64]        -- ^ edge_padding_high
    -> [Int64]        -- ^ interior_padding
    -> Builder (Tensor sOut d)
pad operand paddingValue low high interior = do
    let inType   = tensorType (Proxy @sIn)  (Proxy @d)
        padType  = tensorType (Proxy @'[])  (Proxy @d)
        outType  = tensorType (Proxy @sOut) (Proxy @d)
        lowAttr  = AttrRaw $ "edge_padding_low = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> low) <> ">"
        highAttr = AttrRaw $ "edge_padding_high = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> high) <> ">"
        intAttr  = AttrRaw $ "interior_padding = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> interior) <> ">"

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
             -> [Int64]             -- ^ slice_sizes
             -> Builder (Tensor sOut d)
dynamicSlice operand startIndices sliceSizes = do
    let inType   = tensorType (Proxy @sIn)  (Proxy @d)
        outType  = tensorType (Proxy @sOut) (Proxy @d)
        sizesAttr = AttrRaw $ "slice_sizes = array<i64: " <> T.intercalate ", " ((T.pack . show) <$> sliceSizes) <> ">"

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

    let dimsAttr = AttrRaw $ "dimensions = array<i64: "
            <> T.intercalate ", " ((T.pack . show) <$> dimensions) <> ">"

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
             => [Int64]        -- ^ window_dimensions
             -> [Int64]        -- ^ window_strides
             -> [[Int64]]      -- ^ padding: [[low, high], ...] per dimension
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

    let windowAttr = AttrRaw $ "window_dimensions = array<i64: "
            <> T.intercalate ", " ((T.pack . show) <$> windowDims) <> ">"
        strideAttr = AttrRaw $ "window_strides = array<i64: "
            <> T.intercalate ", " ((T.pack . show) <$> strides) <> ">"
        paddingAttr = AttrRaw $ "padding = dense<[["
            <> T.intercalate "], [" (padPair <$> padding) <> "]]> : tensor<"
            <> T.pack (show (length padding)) <> "x2xi64>"

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
    padPair [l, h] = T.pack (show l) <> ", " <> T.pack (show h)
    padPair _      = error "reduceWindow: padding must be [[low,high], ...]"

-- | 2-D max pooling (NHWC).
maxPool :: forall n h w c oh ow.
           (KnownNat n, KnownNat h, KnownNat w, KnownNat c, KnownNat oh, KnownNat ow)
        => [Int64]   -- ^ kernel [kh, kw]
        -> [Int64]   -- ^ stride [sh, sw]
        -> [[Int64]] -- ^ padding per spatial dim [[pt, pb], [pl, pr]]
        -> Tensor '[n, h, w, c] 'F32
        -> Builder (Tensor '[n, oh, ow, c] 'F32)
maxPool kernel stride padding x = do
    let windowDims = [1, kernel !! 0, kernel !! 1, 1]
        strides    = [1, stride !! 0, stride !! 1, 1]
        fullPadding = [[0, 0], padding !! 0, padding !! 1, [0, 0]]
    -- init value for max: a very negative number
    initVal <- constant @'[] @'F32 (-1.0e30)
    reduceWindow windowDims strides fullPadding "stablehlo.maximum" initVal x

-- | 2-D average pooling (NHWC), VALID padding only.
--
-- Computes the mean over each pooling window. The output size is determined
-- by the input size, kernel, and stride (no padding).
avgPool :: forall n h w c oh ow.
           (KnownNat n, KnownNat h, KnownNat w, KnownNat c, KnownNat oh, KnownNat ow)
        => [Int64]   -- ^ kernel [kh, kw]
        -> [Int64]   -- ^ stride [sh, sw]
        -> Tensor '[n, h, w, c] 'F32
        -> Builder (Tensor '[n, oh, ow, c] 'F32)
avgPool kernel stride x = do
    let windowDims = [1, kernel !! 0, kernel !! 1, 1]
        strides    = [1, stride !! 0, stride !! 1, 1]
        fullPadding = replicate 4 [0, 0]
        windowSize = fromIntegral (product kernel) :: Double
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
                     => [Int64]   -- ^ lhs_dilation (upsample factor), e.g. [1,2,2,1]
                     -> [[Int64]] -- ^ padding per spatial dim, e.g. [[1,1],[1,1]] for 2x2 kernel
                     -> Tensor '[batch, h, w, inCh] 'F32
                     -> Tensor '[kh, kw, outCh, inCh] 'F32
                     -> Builder (Tensor '[batch, oh, ow, outCh] 'F32)
transposeConvolution lhsDilation padding input kernel = do
    let inType1 = tensorType (Proxy @'[batch, h, w, inCh]) (Proxy @'F32)
        inType2 = tensorType (Proxy @'[kh, kw, outCh, inCh]) (Proxy @'F32)
        outType = tensorType (Proxy @'[batch, oh, ow, outCh]) (Proxy @'F32)
        dimNums = "[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]"
        padStr  = "[[" <> T.pack (show (padding !! 0 !! 0)) <> ", " <> T.pack (show (padding !! 0 !! 1)) <> "], ["
               <> T.pack (show (padding !! 1 !! 0)) <> ", " <> T.pack (show (padding !! 1 !! 1)) <> "]]"
        window  = "{stride = [1, 1], pad = " <> padStr
               <> ", lhs_dilate = [" <> T.intercalate ", " ((T.pack . show) <$> drop 1 (take 3 lhsDilation)) <> "]"
               <> ", rhs_dilate = [1, 1]}"
    vid <- emitOp "stablehlo.convolution" [tensorValue input, tensorValue kernel]
            [inType1, inType2]
            [ AttrString "dim_numbers" dimNums
            , AttrString "window" window
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
        (TensorType [fromIntegral (length shapeVals)] I64)
    vid <- emitOp "stablehlo.rng"
            [tensorValue a, tensorValue b, shapeConst]
            [ TensorType [] F32, TensorType [] F32, TensorType [fromIntegral (length shapeVals)] I64 ]
            [AttrRaw "rng_distribution = #stablehlo<rng_distribution UNIFORM>"]
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
        (TensorType [fromIntegral (length shapeVals)] I64)
    vid <- emitOp "stablehlo.rng"
            [tensorValue a, tensorValue b, shapeConst]
            [ TensorType [] F32, TensorType [] F32, TensorType [fromIntegral (length shapeVals)] I64 ]
            [AttrRaw "rng_distribution = #stablehlo<rng_distribution NORMAL>"]
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
              [AttrRaw "rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>"]
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
    sliced <- slice @'[n] @'[1] @d vec [i] [i + 1] [1]
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
