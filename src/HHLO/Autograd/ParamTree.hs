{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DefaultSignatures #-}

module HHLO.Autograd.ParamTree
    ( ParamTree(..)
    , gradWithParams
    ) where

import Data.Int (Int64)
import Data.Proxy
import GHC.Generics
import GHC.TypeLits

import HHLO.Core.Types
import HHLO.IR.AST
import HHLO.IR.Builder

import HHLO.Autograd.Core
import HHLO.Autograd.Grad

-- ---------------------------------------------------------------------------
-- ParamTree: pack/unpack structured parameters
-- ---------------------------------------------------------------------------

-- | A typeclass for structured parameter records that can be packed into a
-- single flat tensor and unpacked back.
--
-- Derive automatically via 'GHC.Generics.Generic':
--
-- > data MLPParams = MLPParams { w :: Tensor '[2,2] 'F32, b :: Tensor '[2] 'F32 }
-- >     deriving (Generic)
-- > instance ParamTree MLPParams
--
-- Only flat records where every field is a 'Tensor' are supported.
class ParamTree a where
    -- | Total number of scalar elements across all tensors.
    paramSize :: Proxy a -> Int

    -- | Element dtype of all tensors (assumed uniform).
    paramDType :: Proxy a -> DType

    -- | Pack a structured record into a single 1-D tensor.
    paramPack :: a -> Builder BTensor

    -- | Unpack a single 1-D tensor back into a structured record.
    paramUnpack :: BTensor -> Builder a

    default paramSize :: (Generic a, GParamTree (Rep a)) => Proxy a -> Int
    paramSize _ = gParamSize (Proxy @(Rep a))

    default paramDType :: (Generic a, GParamTree (Rep a)) => Proxy a -> DType
    paramDType _ = gParamDType (Proxy @(Rep a))

    default paramPack :: (Generic a, GParamTree (Rep a)) => a -> Builder BTensor
    paramPack a = do
        bts <- gParamPack (from a)
        case bts of
            [] -> error "paramPack: empty parameter tree"
            [single] -> return single
            _ -> do
                let totalSize = sum (map (product . ttShape . btType) bts)
                    dtype = ttDType (btType (head bts))
                    resultType = TensorType [fromIntegral totalSize] dtype
                bconcatenate bts 0 resultType

    default paramUnpack :: (Generic a, GParamTree (Rep a)) => BTensor -> Builder a
    paramUnpack bt = do
        (result, _) <- gParamUnpackFrom bt 0
        return (to result)

-- | Base instance for a single tensor.
instance (KnownShape s, KnownDType d) => ParamTree (Tensor s d) where
    paramSize _ = fromIntegral $ product $ shapeVal (Proxy @s)
    paramDType _ = dtypeVal (Proxy @d)
    paramPack tensor = do
        let bt = bfromTyped tensor
            size = product (ttShape (btType bt))
            flatType = TensorType [fromIntegral size] (ttDType (btType bt))
        breshape bt flatType
    paramUnpack bt = do
        let expectedType = tensorType (Proxy @s) (Proxy @d)
        reshaped <- breshape bt expectedType
        return (btoTyped @s @d reshaped)

-- ---------------------------------------------------------------------------
-- Generic derivation via GHC.Generics
-- ---------------------------------------------------------------------------

class GParamTree f where
    gParamSize :: Proxy f -> Int
    gParamDType :: Proxy f -> DType
    gParamPack :: f p -> Builder [BTensor]
    gParamUnpackFrom :: BTensor -> Int -> Builder (f p, Int)

-- Leaf: a single Tensor field.
instance (KnownShape s, KnownDType d) => GParamTree (K1 R (Tensor s d)) where
    gParamSize _ = fromIntegral $ product $ shapeVal (Proxy @s)
    gParamDType _ = dtypeVal (Proxy @d)
    gParamPack (K1 tensor) = do
        bt <- paramPack tensor
        return [bt]
    gParamUnpackFrom flatBt offset = do
        let sizeI = product (shapeVal (Proxy @s))
            size64 = fromIntegral sizeI :: Int64
            expectedType = tensorType (Proxy @s) (Proxy @d)
            sliceType = TensorType [sizeI] (dtypeVal (Proxy @d))
        sliceBt <- bslice flatBt [fromIntegral offset] [fromIntegral offset + size64] [1] sliceType
        reshaped <- breshape sliceBt expectedType
        return (K1 (btoTyped @s @d reshaped), offset + fromIntegral sizeI)

-- Metadata wrapper: transparent.
instance GParamTree f => GParamTree (M1 i c f) where
    gParamSize _ = gParamSize (Proxy @f)
    gParamDType _ = gParamDType (Proxy @f)
    gParamPack (M1 x) = gParamPack x
    gParamUnpackFrom bt off = do
        (x, off') <- gParamUnpackFrom bt off
        return (M1 x, off')

-- Product of two fields: concatenate packs, sequential unpack.
instance (GParamTree f, GParamTree g) => GParamTree (f :*: g) where
    gParamSize _ = gParamSize (Proxy @f) + gParamSize (Proxy @g)
    gParamDType _ = gParamDType (Proxy @f)  -- assumes uniform dtype
    gParamPack (f :*: g) = do
        fs <- gParamPack f
        gs <- gParamPack g
        return (fs ++ gs)
    gParamUnpackFrom bt off = do
        (f, off') <- gParamUnpackFrom bt off
        (g, off'') <- gParamUnpackFrom bt off'
        return (f :*: g, off'')

-- Unit: no fields.
instance GParamTree U1 where
    gParamSize _ = 0
    gParamDType _ = F32
    gParamPack U1 = return []
    gParamUnpackFrom _ off = return (U1, off)

-- ---------------------------------------------------------------------------
-- gradWithParams: ergonomic multi-parameter gradient
-- ---------------------------------------------------------------------------

-- | Compute gradients of a scalar-valued loss w.r.t. a structured parameter
-- record.
--
-- This hides the pack/unpack boilerplate entirely.  The user writes the
-- forward pass with a structured parameter record, and receives a structured
-- gradient record of the same shape.
--
-- Example:
--
-- > data MLPParams = MLPParams { w :: Tensor '[2,2] 'F32, b :: Tensor '[2] 'F32 }
-- >     deriving (Generic)
-- > instance ParamTree MLPParams
-- >
-- > loss params x = do
-- >     y <- add (matmul x (w params)) (b params)
-- >     diff <- sub y target
-- >     sumAll (multiply diff diff)
-- >
-- > trainStep params x = gradWithParams loss params x
--
-- All parameters must have the same dtype as the loss.
gradWithParams :: forall p s d.
                  (ParamTree p, KnownShape s, KnownDType d)
               => (p -> Tensor s d -> Builder (Tensor '[] d))
               -> p -> Tensor s d
               -> Builder p
gradWithParams lossFn params input = do
    packed <- paramPack params
    let n = paramSize (Proxy @p)
        dt = paramDType (Proxy @p)
    if dt /= dtypeVal (Proxy @d)
        then error $ "gradWithParams: parameter dtype " ++ show dt ++
                     " does not match loss dtype " ++ show (dtypeVal (Proxy @d))
        else reifyShape [fromIntegral n] $ \(_ :: Proxy nShape) -> do
            let packedLoss :: Tensor nShape d -> Builder (Tensor '[] d)
                packedLoss packedTensor = do
                    params' <- paramUnpack (bfromTyped packedTensor)
                    lossFn params' input
            dPacked <- grad packedLoss (btoTyped @nShape @d packed)
            paramUnpack (bfromTyped dPacked)
