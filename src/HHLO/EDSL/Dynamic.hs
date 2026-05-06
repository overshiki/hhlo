{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Dynamic-shape operations for HHLO.
--
-- This module provides an escape hatch for programs where tensor shapes
-- are not known at Haskell compile time.  It is fully additive: the
-- static 'Tensor' API is unchanged.
--
-- Example (dynamic reshape + add):
--
-- > moduleFromBuilderDynamic "main"
-- >     [ FuncArg "a" (TensorType [Just 3, Nothing] F32) ]
-- >     (TensorType [Just 3, Nothing] F32)
-- >   $ do
-- >       a <- anyArg (TensorType [Just 3, Nothing] F32)
-- >       b <- anyDynamicReshape a [Just 1, Just 3, Nothing]
-- >       return (anyVid b)
--
module HHLO.EDSL.Dynamic
    ( -- * Dynamic tensor type
      AnyTensor(..)
      -- * Module builders
    , moduleFromBuilderDynamic
      -- * Argument declarations
    , anyArg
    , anyArgNamed
      -- * Shape-changing ops
    , anyDynamicReshape
    , anyGetDimensionSize
    , anySetDimensionSize
      -- * Slicing ops
    , anyDynamicSlice
    , anyDynamicUpdateSlice
      -- * Elementwise ops
    , anyAdd
    , anySubtract
    , anyMultiply
    , anyDivide
      -- * Conversions
    , anyFromTyped
    , anyToTyped
    ) where

import Control.Monad (when)
import Control.Monad.State
import Data.Int (Int64)
import Data.Maybe (fromJust, isNothing)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T

import HHLO.Core.Types
import HHLO.IR.AST
import HHLO.IR.Builder
import HHLO.ShapeCheck (shapeMatch)

-- | An untyped tensor for dynamic-shape programming.
-- Carries its full 'TensorType' (including dynamic dimensions) at runtime.
data AnyTensor (d :: DType) = AnyTensor
    { anyVid  :: ValueId
    , anyType :: TensorType
    }

-- | Declare a dynamically-shaped function argument.
anyArg :: TensorType -> Builder (AnyTensor d)
anyArg ttype = do
    n <- gets bsArgCount
    modify $ \s -> s { bsArgCount = n + 1 }
    return $ AnyTensor (ValueId (-(n + 1))) ttype

-- | Declare a dynamically-shaped function argument with a specific name.
anyArgNamed :: Text -> TensorType -> Builder (AnyTensor d)
anyArgNamed _name = anyArg

-- | Wrap a statically-typed tensor into an 'AnyTensor'.
anyFromTyped :: forall s d. (KnownShape s, KnownDType d) => Tensor s d -> AnyTensor d
anyFromTyped (Tensor vid) = AnyTensor vid (tensorType (Proxy @s) (Proxy @d))

-- | Unsafely cast an 'AnyTensor' to a statically-typed 'Tensor'.
-- The caller is responsible for ensuring the runtime shape matches
-- the type-level shape.
anyToTyped :: forall s d. AnyTensor d -> Tensor s d
anyToTyped (AnyTensor vid _) = Tensor vid

-- | Reshape a tensor to a new shape where some dimensions may be dynamic.
-- Dynamic dimensions in the output shape are represented by 'Nothing';
-- XLA infers them from the total element count.
anyDynamicReshape :: AnyTensor d -> [Maybe Integer] -> Builder (AnyTensor d)
anyDynamicReshape (AnyTensor vid ttype) outShape = do
    -- Build a constant shape tensor: -1 for dynamic dims, literal for static dims.
    let shapeVals = [if isNothing m then -1 else fromIntegral (fromJust m) | m <- outShape] :: [Int64]
        nDims     = fromIntegral (length outShape) :: Integer
        shapeType = TensorType [Just nDims] I64
    shapeConst <- emitOp "stablehlo.constant" [] []
        [AttrDenseElements [nDims] I64 (map fromIntegral shapeVals)]
        shapeType
    resVid <- emitDynamicReshape vid ttype shapeConst shapeType outShape
    return $ AnyTensor resVid (TensorType outShape (ttDType ttype))

-- | Query the size of a dimension at runtime.
-- Returns a scalar 'i32' tensor.
anyGetDimensionSize :: AnyTensor d -> Int -> Builder (Tensor '[] 'I32)
anyGetDimensionSize (AnyTensor vid ttype) dim = do
    resVid <- emitGetDimensionSize vid ttype dim
    return (Tensor resVid)

-- | Set the size of a dimension, turning it into a dynamic dimension.
anySetDimensionSize :: AnyTensor d -> Tensor '[] 'I64 -> Int -> Builder (AnyTensor d)
anySetDimensionSize (AnyTensor vid ttype) (Tensor sizeVid) dim = do
    let sizeType = TensorType [] I64
    resVid <- emitSetDimensionSize vid ttype sizeVid sizeType dim
    let outShape = take dim (ttShape ttype) ++ [Nothing] ++ drop (dim + 1) (ttShape ttype)
    return $ AnyTensor resVid (TensorType outShape (ttDType ttype))

-- | Extract a slice with runtime start indices.
anyDynamicSlice :: AnyTensor d -> [Tensor '[] 'I64] -> [Int] -> Builder (AnyTensor d)
anyDynamicSlice (AnyTensor vid ttype) startIndices sliceSizes = do
    let outShape = map (Just . fromIntegral) sliceSizes
        outType  = TensorType outShape (ttDType ttype)
        sizesAttr = AttrIntList "slice_sizes" (map fromIntegral sliceSizes)
        startVids = map tensorValue startIndices
        startTypes = replicate (length startIndices) (TensorType [] I64)
    resVid <- emitOp "stablehlo.dynamic_slice"
        (vid : startVids)
        (ttype : startTypes)
        [sizesAttr]
        outType
    return $ AnyTensor resVid outType

-- | Generic binary elementwise operation on two 'AnyTensor's.
-- The shapes must match (treating 'Nothing' as a wildcard).
anyBinaryOp :: Text -> AnyTensor d -> AnyTensor d -> Builder (AnyTensor d)
anyBinaryOp opName (AnyTensor vid1 t1) (AnyTensor vid2 t2) = do
    when (ttDType t1 /= ttDType t2) $
        error $ "anyBinaryOp: dtype mismatch: " ++ show (ttDType t1) ++ " vs " ++ show (ttDType t2)
    when (not (shapeMatch (ttShape t1) (ttShape t2))) $
        error $ "anyBinaryOp: shape mismatch between " ++ show (ttShape t1) ++ " and " ++ show (ttShape t2)
    resVid <- emitOp opName [vid1, vid2] [t1, t2] [] t1
    return $ AnyTensor resVid t1

-- | Elementwise addition.
anyAdd :: AnyTensor d -> AnyTensor d -> Builder (AnyTensor d)
anyAdd = anyBinaryOp "stablehlo.add"

-- | Elementwise subtraction.
anySubtract :: AnyTensor d -> AnyTensor d -> Builder (AnyTensor d)
anySubtract = anyBinaryOp "stablehlo.subtract"

-- | Elementwise multiplication.
anyMultiply :: AnyTensor d -> AnyTensor d -> Builder (AnyTensor d)
anyMultiply = anyBinaryOp "stablehlo.multiply"

-- | Elementwise division.
anyDivide :: AnyTensor d -> AnyTensor d -> Builder (AnyTensor d)
anyDivide = anyBinaryOp "stablehlo.divide"

-- | Update a slice with runtime start indices.
anyDynamicUpdateSlice :: AnyTensor d -> AnyTensor d -> [Tensor '[] 'I64] -> Builder (AnyTensor d)
anyDynamicUpdateSlice (AnyTensor vid ttype) (AnyTensor updateVid updateType) startIndices = do
    let startVids = map tensorValue startIndices
        startTypes = replicate (length startIndices) (TensorType [] I64)
    resVid <- emitDynamicUpdateSlice vid ttype updateVid updateType startVids startTypes
    return $ AnyTensor resVid ttype
