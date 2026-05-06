{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.Autograd.Core
    ( BTensor(..)
    , CotangentMap
    , bconstant
    , badd
    , bsubtract
    , bmultiply
    , bdivide
    , bnegate
    , bexp
    , blog
    , bsqrt
    , bsin
    , bcos
    , btranspose
    , breshape
    , bbroadcastInDim
    , breduceSum
    , bdot
    , bdotGeneral
    , babs
    , btanh
    , bmaximum
    , bminimum
    , bselect
    , bslice
    , bpad
    , bconcatenate
    , bconvert
    , bcompareGE
    , bcompareEQ
    , btoTyped
    , bfromTyped
    , reifyShape
    , accumulate
    , breverse
    , bconvolution
    , breduceWindowAdd
    , breduceWindowMax
    ) where

import Data.Int (Int64)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import GHC.TypeLits
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

import HHLO.Core.Types
import HHLO.IR.AST
import HHLO.IR.Builder

-- ---------------------------------------------------------------------------
-- Backward tensor: runtime-typed tensor handle for the backward pass.
-- ---------------------------------------------------------------------------

-- | A tensor value used during the backward pass, carrying only runtime
-- shape and dtype information.
data BTensor = BTensor
    { btVid  :: !ValueId
    , btType :: !TensorType
    } deriving (Eq, Show)

-- | Map from forward value ID to its accumulated cotangent.
type CotangentMap = Map ValueId BTensor

-- ---------------------------------------------------------------------------
-- Existential shape reification
-- ---------------------------------------------------------------------------

-- | Reify a runtime shape list into a type-level 'KnownShape' constraint.
reifyShape :: [Integer] -> (forall (s :: Shape). KnownShape s => Proxy s -> r) -> r
reifyShape [] k = k (Proxy @'[])
reifyShape (n : ns) k =
    case someNatVal (fromIntegral n) of
        Just (SomeNat (_ :: Proxy n)) ->
            reifyShape ns $ \(_ :: Proxy ns) ->
                k (Proxy @(n ': ns))
        Nothing -> error "autograd-hhlo: negative dimension in reifyShape"

-- ---------------------------------------------------------------------------
-- Conversions between BTensor and typed Tensor
-- ---------------------------------------------------------------------------

-- | Convert a typed 'Tensor' to a runtime-typed 'BTensor'.
bfromTyped :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> BTensor
bfromTyped (Tensor vid) = BTensor vid (tensorType (Proxy @s) (Proxy @d))

-- | Convert a 'BTensor' to a typed 'Tensor', checking that the runtime
-- type matches the expected compile-time type.
btoTyped :: forall s d. (KnownShape s, KnownDType d) => BTensor -> Tensor s d
btoTyped (BTensor vid ttype) =
    let expected = tensorType (Proxy @s) (Proxy @d)
    in if ttype == expected
        then Tensor vid
        else error $ T.unpack $
            "autograd-hhlo: type mismatch in btoTyped. Expected "
            <> T.pack (show expected) <> ", got " <> T.pack (show ttype)

-- ---------------------------------------------------------------------------
-- Cotangent accumulation
-- ---------------------------------------------------------------------------

-- | Add a new cotangent to an existing entry in the map, or insert it
-- if none exists yet.
accumulate :: CotangentMap -> ValueId -> BTensor -> Builder CotangentMap
accumulate cmap vid newBar =
    case Map.lookup vid cmap of
        Just existingBar -> do
            summed <- badd existingBar newBar
            return $! Map.insert vid summed cmap
        Nothing ->
            return $! Map.insert vid newBar cmap

-- ---------------------------------------------------------------------------
-- Primitive backward operations (emit raw StableHLO)
-- ---------------------------------------------------------------------------

bconstant :: TensorType -> Double -> Builder BTensor
bconstant t val = do
    let shp = ttShape t
        dt  = ttDType t
        numElems = product [x | Just x <- shp]
    vid <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [x | Just x <- shp] dt (replicate (fromIntegral numElems) val)] t
    return (BTensor vid t)

badd :: BTensor -> BTensor -> Builder BTensor
badd (BTensor x t) (BTensor y _) = do
    vid <- emitOp "stablehlo.add" [x, y] [t, t] [] t
    return (BTensor vid t)

bsubtract :: BTensor -> BTensor -> Builder BTensor
bsubtract (BTensor x t) (BTensor y _) = do
    vid <- emitOp "stablehlo.subtract" [x, y] [t, t] [] t
    return (BTensor vid t)

bmultiply :: BTensor -> BTensor -> Builder BTensor
bmultiply (BTensor x t) (BTensor y _) = do
    vid <- emitOp "stablehlo.multiply" [x, y] [t, t] [] t
    return (BTensor vid t)

bdivide :: BTensor -> BTensor -> Builder BTensor
bdivide (BTensor x t) (BTensor y _) = do
    vid <- emitOp "stablehlo.divide" [x, y] [t, t] [] t
    return (BTensor vid t)

bnegate :: BTensor -> Builder BTensor
bnegate (BTensor x t) = do
    vid <- emitOp "stablehlo.negate" [x] [t] [] t
    return (BTensor vid t)

bexp :: BTensor -> Builder BTensor
bexp (BTensor x t) = do
    vid <- emitOp "stablehlo.exponential" [x] [t] [] t
    return (BTensor vid t)

blog :: BTensor -> Builder BTensor
blog (BTensor x t) = do
    vid <- emitOp "stablehlo.log" [x] [t] [] t
    return (BTensor vid t)

bsqrt :: BTensor -> Builder BTensor
bsqrt (BTensor x t) = do
    vid <- emitOp "stablehlo.sqrt" [x] [t] [] t
    return (BTensor vid t)

bsin :: BTensor -> Builder BTensor
bsin (BTensor x t) = do
    vid <- emitOp "stablehlo.sine" [x] [t] [] t
    return (BTensor vid t)

bcos :: BTensor -> Builder BTensor
bcos (BTensor x t) = do
    vid <- emitOp "stablehlo.cosine" [x] [t] [] t
    return (BTensor vid t)

-- | Transpose a BTensor using a permutation.
btranspose :: BTensor -> [Int64] -> TensorType -> Builder BTensor
btranspose (BTensor x inType) perm outType = do
    let permAttr = AttrIntList "permutation" perm
    vid <- emitOp "stablehlo.transpose" [x] [inType] [permAttr] outType
    return (BTensor vid outType)

-- | Reshape a BTensor.
breshape :: BTensor -> TensorType -> Builder BTensor
breshape (BTensor x inType) outType = do
    vid <- emitOp "stablehlo.reshape" [x] [inType] [] outType
    return (BTensor vid outType)

-- | Broadcast a BTensor to a new shape with explicit dimension mapping.
bbroadcastInDim :: BTensor -> [Int64] -> TensorType -> Builder BTensor
bbroadcastInDim (BTensor x inType) dims outType = do
    vid <- emitOp "stablehlo.broadcast_in_dim" [x] [inType]
        [AttrIntList "broadcast_dimensions" (fromIntegral <$> dims)] outType
    return (BTensor vid outType)

-- | Element-wise selection between two BTensors based on a boolean predicate.
bselect :: BTensor -> BTensor -> BTensor -> TensorType -> Builder BTensor
bselect (BTensor p predType) (BTensor t _) (BTensor f _) outType = do
    vid <- emitOp "stablehlo.select" [p, t, f] [predType, outType, outType] [] outType
    return (BTensor vid outType)

-- | Slice a BTensor (forward operation wrapper).
bslice :: BTensor -> [Int64] -> [Int64] -> [Int64] -> TensorType -> Builder BTensor
bslice (BTensor x inType) start limit stride outType = do
    let startAttr  = AttrIntList "start_indices" start
        limitAttr  = AttrIntList "limit_indices" limit
        strideAttr = AttrIntList "strides" stride
    vid <- emitOp "stablehlo.slice" [x] [inType] [startAttr, limitAttr, strideAttr] outType
    return (BTensor vid outType)

-- | Pad a BTensor with edge and interior padding.
bpad :: BTensor -> BTensor -> [Int64] -> [Int64] -> [Int64] -> TensorType -> Builder BTensor
bpad (BTensor x inType) (BTensor padVal padType) low high interior outType = do
    let lowAttr  = AttrIntList "edge_padding_low" low
        highAttr = AttrIntList "edge_padding_high" high
        intAttr  = AttrIntList "interior_padding" interior
    vid <- emitOp "stablehlo.pad" [x, padVal] [inType, padType] [lowAttr, highAttr, intAttr] outType
    return (BTensor vid outType)

-- | Concatenate multiple BTensors along a dimension.
bconcatenate :: [BTensor] -> Int64 -> TensorType -> Builder BTensor
bconcatenate inputs dim outType = do
    let vids = map btVid inputs
        types = map btType inputs
        dimAttr = AttrInt "dimension" (fromIntegral dim)
    vid <- emitOp "stablehlo.concatenate" vids types [dimAttr] outType
    return (BTensor vid outType)

-- | Reduce-sum a BTensor over specified dimensions.
breduceSum :: BTensor -> [Int] -> TensorType -> Builder BTensor
breduceSum (BTensor x inType) dims outType = do
    let elemType = TensorType [] (ttDType inType)
    zeroVid <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [] (ttDType inType) [0.0]] elemType
    redBlock <- runBlockBuilder [elemType, elemType] $ do
        a <- arg @'[] @( 'F32)
        b <- arg @'[] @( 'F32)
        sumVid <- emitOp "stablehlo.add"
                    [tensorValue a, tensorValue b]
                    [elemType, elemType] [] elemType
        emitReturn [sumVid] [elemType]
    vid <- emitOpRegions "stablehlo.reduce"
            [x, zeroVid]
            [inType, elemType]
            [AttrIntList "dimensions" (map fromIntegral dims)]
            [Region [redBlock]]
            outType
    return (BTensor vid outType)

-- | Matrix multiplication of two BTensors.
bdot :: BTensor -> BTensor -> TensorType -> Builder BTensor
bdot (BTensor x t1) (BTensor y t2) outType = do
    vid <- emitOp "stablehlo.dot" [x, y] [t1, t2] [] outType
    return (BTensor vid outType)

-- | General dot product of two BTensors with explicit dimension numbers.
bdotGeneral :: BTensor -> BTensor -> [Int64] -> [Int64] -> [Int64] -> [Int64] -> TensorType -> Builder BTensor
bdotGeneral (BTensor lhs lhsType) (BTensor rhs rhsType) batchL batchR contractL contractR outType = do
    let attrs =
            [ AttrIntList "lhs_batching_dimensions" batchL
            , AttrIntList "rhs_batching_dimensions" batchR
            , AttrIntList "lhs_contracting_dimensions" contractL
            , AttrIntList "rhs_contracting_dimensions" contractR
            ]
    vid <- emitOp "stablehlo.dot_general" [lhs, rhs] [lhsType, rhsType] attrs outType
    return (BTensor vid outType)

-- | Element-wise absolute value.
babs :: BTensor -> Builder BTensor
babs (BTensor x t) = do
    vid <- emitOp "stablehlo.abs" [x] [t] [] t
    return (BTensor vid t)

-- | Element-wise hyperbolic tangent.
btanh :: BTensor -> Builder BTensor
btanh (BTensor x t) = do
    vid <- emitOp "stablehlo.tanh" [x] [t] [] t
    return (BTensor vid t)

-- | Element-wise maximum.
bmaximum :: BTensor -> BTensor -> Builder BTensor
bmaximum (BTensor x t1) (BTensor y t2) = do
    vid <- emitOp "stablehlo.maximum" [x, y] [t1, t2] [] t1
    return (BTensor vid t1)

-- | Element-wise minimum.
bminimum :: BTensor -> BTensor -> Builder BTensor
bminimum (BTensor x t1) (BTensor y t2) = do
    vid <- emitOp "stablehlo.minimum" [x, y] [t1, t2] [] t1
    return (BTensor vid t1)

-- | Type conversion.
bconvert :: BTensor -> TensorType -> Builder BTensor
bconvert (BTensor x inType) outType = do
    vid <- emitOp "stablehlo.convert" [x] [inType] [] outType
    return (BTensor vid outType)

-- | Greater-than-or-equal comparison (returns boolean tensor).
bcompareGE :: BTensor -> BTensor -> Builder BTensor
bcompareGE (BTensor x t1) (BTensor y t2) = do
    let boolType = TensorType (ttShape t1) Bool
    vid <- emitOp "stablehlo.compare" [x, y] [t1, t2]
        [AttrRaw "comparison_direction = #stablehlo<comparison_direction GE>"] boolType
    return (BTensor vid boolType)

-- | Equal comparison (returns boolean tensor).
bcompareEQ :: BTensor -> BTensor -> Builder BTensor
bcompareEQ (BTensor x t1) (BTensor y t2) = do
    let boolType = TensorType (ttShape t1) Bool
    vid <- emitOp "stablehlo.compare" [x, y] [t1, t2]
        [AttrRaw "comparison_direction = #stablehlo<comparison_direction EQ>"] boolType
    return (BTensor vid boolType)

-- | Reverse a tensor along specified dimensions.
breverse :: BTensor -> [Int64] -> TensorType -> Builder BTensor
breverse (BTensor x t) dims outType = do
    let dimsAttr = AttrIntList "dimensions" dims
    vid <- emitOp "stablehlo.reverse" [x] [t] [dimsAttr] outType
    return (BTensor vid outType)

-- | Generic convolution emitter.
--
-- This is the low-level primitive used by VJP rules.  It emits a
-- @stablehlo.convolution@ with fully-specified structured attributes
-- (dimension numbers and window attributes).
bconvolution :: BTensor -> BTensor -> [Attribute] -> TensorType -> Builder BTensor
bconvolution (BTensor lhs lhsType) (BTensor rhs rhsType) attrs outType = do
    vid <- emitOp "stablehlo.convolution"
            [lhs, rhs]
            [lhsType, rhsType]
            attrs
            outType
    return (BTensor vid outType)

-- | Reduce a BTensor over specified window dimensions with @add@.
breduceWindowAdd :: BTensor -> BTensor -> [Int64] -> [Int64] -> [[Int64]] -> TensorType -> Builder BTensor
breduceWindowAdd (BTensor input inType) (BTensor initVal initType) windowDims strides padding outType = do
    let elemType = TensorType [] (ttDType inType)
    -- Build the add reduction region.
    redBlock <- runBlockBuilder [elemType, elemType] $ do
        a <- arg @'[] @( 'F32)
        b <- arg @'[] @( 'F32)
        sumVid <- emitOp "stablehlo.add"
                    [tensorValue a, tensorValue b]
                    [elemType, elemType] [] elemType
        emitReturn [sumVid] [elemType]

    let windowAttr  = AttrIntList "window_dimensions" windowDims
        strideAttr  = AttrIntList "window_strides" strides
        paddingAttr = AttrRaw $ "padding = dense<[["
            <> T.intercalate "], [" (padPair <$> padding) <> "]]> : tensor<"
            <> T.pack (show (length padding)) <> "x2xi64>"

    vid <- emitOpRegions "stablehlo.reduce_window"
            [input, initVal]
            [inType, initType]
            [windowAttr, strideAttr, paddingAttr]
            [Region [redBlock]]
            outType
    return (BTensor vid outType)
  where
    padPair [l, h] = T.pack (show l) <> ", " <> T.pack (show h)
    padPair _      = error "breduceWindowAdd: padding must be [[low,high], ...]"

-- | Reduce a BTensor over specified window dimensions with @maximum@.
breduceWindowMax :: BTensor -> BTensor -> [Int64] -> [Int64] -> [[Int64]] -> TensorType -> Builder BTensor
breduceWindowMax (BTensor input inType) (BTensor initVal initType) windowDims strides padding outType = do
    let elemType = TensorType [] (ttDType inType)
    -- Build the max reduction region.
    redBlock <- runBlockBuilder [elemType, elemType] $ do
        a <- arg @'[] @( 'F32)
        b <- arg @'[] @( 'F32)
        maxVid <- emitOp "stablehlo.maximum"
                    [tensorValue a, tensorValue b]
                    [elemType, elemType] [] elemType
        emitReturn [maxVid] [elemType]

    let windowAttr  = AttrIntList "window_dimensions" windowDims
        strideAttr  = AttrIntList "window_strides" strides
        paddingAttr = AttrRaw $ "padding = dense<[["
            <> T.intercalate "], [" (padPair <$> padding) <> "]]> : tensor<"
            <> T.pack (show (length padding)) <> "x2xi64>"

    vid <- emitOpRegions "stablehlo.reduce_window"
            [input, initVal]
            [inType, initType]
            [windowAttr, strideAttr, paddingAttr]
            [Region [redBlock]]
            outType
    return (BTensor vid outType)
  where
    padPair [l, h] = T.pack (show l) <> ", " <> T.pack (show h)
    padPair _      = error "breduceWindowMax: padding must be [[low,high], ...]"
