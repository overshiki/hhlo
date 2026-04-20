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
    , reduceSumDim
    -- * Neural network layers
    , softmax1D
    , softmax2D
    , conv2d
    , batchNormInference
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
    let inType  = tensorType (Proxy @sFrom) (Proxy @d)
        outType = tensorType (Proxy @sTo)   (Proxy @d)
        scalarType = tensorType (Proxy @'[]) (Proxy @d)
    -- The init value for stablehlo.reduce must be a scalar, even when the
    -- result is non-scalar.  The scalar is used as the initial value for
    -- each independent reduction.
    zeroVid <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [] (dtypeVal (Proxy @d)) [0.0]] scalarType
    vid <- emitReduce x inType zeroVid scalarType dims "stablehlo.add" outType
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
conv2d (Tensor x) (Tensor k) = do
    let inType1 = tensorType (Proxy @'[batch, h, w, inCh])   (Proxy @'F32)
        inType2 = tensorType (Proxy @'[kh, kw, inCh, outCh]) (Proxy @'F32)
        outType = tensorType (Proxy @'[batch, oh, ow, outCh]) (Proxy @'F32)
        dimNums = "[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]"
        window  = "{pad = [[0, 0], [0, 0]]}"
    vid <- emitOp "stablehlo.convolution" [x, k] [inType1, inType2]
        [ AttrString "dim_numbers" dimNums
        , AttrString "window" window
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

-- | Return a heterogeneous tuple of tensors from a multi-result builder.
returnT :: Tuple ss ds -> Builder (Tuple ss ds)
returnT = return
