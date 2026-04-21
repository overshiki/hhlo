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
    -- * Control flow
    , whileLoop
    , conditional
    , compare
    , lessThan
    -- * Data movement
    , gather
    , scatter
    , slice
    , pad
    , dynamicSlice
    , sort
    , convert
    -- * Constants
    , constant
    -- * Tuple
    , Tuple2(..)
    , returnTuple2
    , Tuple(..)
    , returnT
    ) where

import Prelude hiding (subtract, negate, maximum, minimum, abs, compare)

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
        => Tensor s d -> Tensor s d -> Text -> Builder (Tensor '[] 'Bool)
compare (Tensor x) (Tensor y) direction = do
    let inType  = tensorType (Proxy @s) (Proxy @d)
        outType = tensorType (Proxy @'[]) (Proxy @'Bool)
    vid <- emitOp "stablehlo.compare" [x, y] [inType, inType]
        [ AttrString "comparison_direction" direction
        ] outType
    return (Tensor vid)

-- | Convenience wrapper for 'compare' with @"LT"@ direction.
lessThan :: forall s d.
            (KnownShape s, KnownDType d)
         => Tensor s d -> Tensor s d -> Builder (Tensor '[] 'Bool)
lessThan x y = compare x y "LT"

-- ---------------------------------------------------------------------------
-- Data movement
-- ---------------------------------------------------------------------------

-- | Helper: format an integer list for MLIR attribute text.
intList :: [Int64] -> Text
intList xs = "[" <> T.intercalate ", " (map (T.pack . show) xs) <> "]"

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
            "slice_sizes = array<i64: " <> T.intercalate ", " (map (T.pack . show) sliceSizes) <> ">"
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
        startAttr = AttrRaw $ "start_indices = array<i64: " <> T.intercalate ", " (map (T.pack . show) start) <> ">"
        limitAttr = AttrRaw $ "limit_indices = array<i64: " <> T.intercalate ", " (map (T.pack . show) limit) <> ">"
        strideAttr = AttrRaw $ "strides = array<i64: " <> T.intercalate ", " (map (T.pack . show) stride) <> ">"

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
        lowAttr  = AttrRaw $ "edge_padding_low = array<i64: " <> T.intercalate ", " (map (T.pack . show) low) <> ">"
        highAttr = AttrRaw $ "edge_padding_high = array<i64: " <> T.intercalate ", " (map (T.pack . show) high) <> ">"
        intAttr  = AttrRaw $ "interior_padding = array<i64: " <> T.intercalate ", " (map (T.pack . show) interior) <> ">"

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
        sizesAttr = AttrRaw $ "slice_sizes = array<i64: " <> T.intercalate ", " (map (T.pack . show) sliceSizes) <> ">"

    let (Tensor operandVid) = operand
        startVids = map tensorValue startIndices
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
