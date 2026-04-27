{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.Autograd.Grad
    ( grad
    , gradModule
    , vjp
    , vjpModule
    ) where

import Data.List (foldl')
import Data.Proxy
import Data.Foldable (foldlM)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import HHLO.Core.Types
import HHLO.EDSL.Ops (constant)
import HHLO.IR.AST
import HHLO.IR.Builder

import HHLO.Autograd.Core
import HHLO.Autograd.Rules

-- ---------------------------------------------------------------------------
-- Safe API: gradModule / vjpModule
-- ---------------------------------------------------------------------------

-- | Compute the gradient of a scalar-valued function as a standalone
-- 'Module'.
--
-- The resulting module takes a single tensor argument and returns its
-- gradient (a tensor of the same shape).
gradModule :: forall s d.
              (KnownShape s, KnownDType d)
           => (Tensor s d -> Builder (Tensor '[] d))
           -> Module
gradModule f = Module [gradFunction]
  where
    inType = tensorType (Proxy @s) (Proxy @d)
    arg0   = FuncArg "arg0" inType

    -- Capture the forward trace in isolation.
    forwardFunc :: Function
    forwardFunc = runBuilder @'[] @d "forward" [arg0] $ do
        input <- arg @s @d
        f input

    forwardOps :: [Operation]
    forwardOps = funcBody forwardFunc

    -- Build the combined forward+backward function.
    gradFunction :: Function
    gradFunction = runBuilder @s @d "main" [arg0] $ do
        input <- arg @s @d
        y     <- f input

        -- Seed cotangent: 1.0 scalar.
        seed <- constant @'[] @d 1.0
        let initMap = Map.singleton (tensorValue y) (bfromTyped seed)

        -- Backpropagate through forward ops in reverse execution order.
        finalMap <- foldlBackward backwardStep initMap (reverse forwardOps)

        -- Extract gradient w.r.t. input.
        case Map.lookup (tensorValue input) finalMap of
            Just bt  -> return (btoTyped @s bt)
            Nothing  -> error "autograd-hhlo: gradient not found for input"

-- | Vector-Jacobian product as a standalone 'Module'.
vjpModule :: forall s t d.
             (KnownShape s, KnownShape t, KnownDType d)
          => (Tensor s d -> Builder (Tensor t d))
          -> Module
vjpModule f = Module [vjpFunction]
  where
    inType  = tensorType (Proxy @s) (Proxy @d)
    seedType = tensorType (Proxy @t) (Proxy @d)
    arg0 = FuncArg "arg0" inType
    arg1 = FuncArg "arg1" seedType

    forwardFunc :: Function
    forwardFunc = runBuilder @t @d "forward" [arg0] $ do
        input <- arg @s @d
        f input

    forwardOps :: [Operation]
    forwardOps = funcBody forwardFunc

    vjpFunction :: Function
    vjpFunction = runBuilder @s @d "main" [arg0, arg1] $ do
        input <- arg @s @d
        seed  <- arg @t @d
        _y    <- f input

        let initMap = Map.singleton (tensorValue _y) (bfromTyped seed)
        finalMap <- foldlBackward backwardStep initMap (reverse forwardOps)

        case Map.lookup (tensorValue input) finalMap of
            Just bt -> return (btoTyped @s bt)
            Nothing -> error "autograd-hhlo: vjp gradient not found for input"

-- ---------------------------------------------------------------------------
-- In-place API: grad / vjp
-- ---------------------------------------------------------------------------

-- | Reverse-mode gradient combinator that works inside an existing
-- 'Builder' context.
--
-- This builds the gradient module in isolation and inlines its operations
-- into the current builder, remapping value IDs so they remain globally
-- consistent.
--
-- Example:
-- > buildModule @1 @1 "grad_f" $ \x -> grad (\y -> sumAll (multiply y y)) x
grad :: forall s d.
        (KnownShape s, KnownDType d)
     => (Tensor s d -> Builder (Tensor '[] d))
     -> Tensor s d
     -> Builder (Tensor s d)
grad f input = do
    let gradMod = gradModule f
        gradFunc = head (moduleFunctions gradMod)
    gradVid <- inlineFunction gradFunc (Map.singleton (ValueId (-1)) (tensorValue input))
    return (Tensor gradVid)

-- | Vector-Jacobian product inside an existing 'Builder' context.
vjp :: forall s t d.
       (KnownShape s, KnownShape t, KnownDType d)
    => (Tensor s d -> Builder (Tensor t d))
    -> Tensor s d
    -> Tensor t d
    -> Builder (Tensor s d)
vjp f input seed = do
    let vjpMod = vjpModule f
        vjpFunc = head (moduleFunctions vjpMod)
    gradVid <- inlineFunction vjpFunc $ Map.fromList
        [ (ValueId (-1), tensorValue input)
        , (ValueId (-2), tensorValue seed)
        ]
    return (Tensor gradVid)

-- ---------------------------------------------------------------------------
-- Internal helpers
-- ---------------------------------------------------------------------------

-- | Fold a backward step over a list of forward operations, threading the
-- cotangent map through the Builder monad.
foldlBackward :: (Map ValueId BTensor -> Operation -> Builder (Map ValueId BTensor))
              -> Map ValueId BTensor
              -> [Operation]
              -> Builder (Map ValueId BTensor)
foldlBackward _ cmap [] = return cmap
foldlBackward step cmap (op:ops) = do
    cmap' <- step cmap op
    foldlBackward step cmap' ops

-- | Inline a function's body operations into the current builder,
-- remapping value IDs so they don't conflict with existing values.
-- The argument mapping maps function argument IDs (negative) to the
-- current builder's actual value IDs.
inlineFunction :: Function -> Map ValueId ValueId -> Builder ValueId
inlineFunction func argMap = do
    finalMap <- foldlM inlineOp argMap (funcBody func)
    let retId = head (funcReturnVids func)
    return $ Map.findWithDefault retId retId finalMap
  where
    inlineOp :: Map ValueId ValueId -> Operation -> Builder (Map ValueId ValueId)
    inlineOp idMap op = do
        let remap vid = Map.findWithDefault vid vid idMap
            newOperands = map remap (opOperands op)
            newRegions = map (remapRegion idMap) (opRegions op)
        newVids <- emitOpRegionsN (opName op) newOperands (opOperandTypes op) (opAttributes op) newRegions (opResultTypes op)
        let updates = zip (opResults op) newVids
        return $ foldl' (\m (old, new) -> Map.insert old new m) idMap updates

    remapRegion :: Map ValueId ValueId -> Region -> Region
    remapRegion idMap (Region blocks) = Region (map (remapBlock idMap) blocks)

    remapBlock :: Map ValueId ValueId -> Block -> Block
    remapBlock idMap (Block blockArgs blockOps) =
        Block blockArgs (map (remapOp idMap) blockOps)

    remapOp :: Map ValueId ValueId -> Operation -> Operation
    remapOp idMap op =
        let remapOpVid vid = Map.findWithDefault vid vid idMap
        in op
            { opOperands = map remapOpVid (opOperands op)
            , opResults  = map remapOpVid (opResults op)
            }

    -- Note: block arguments are local to the block and use negative IDs
    -- allocated by runBlockBuilder. They are self-contained and don't
    -- need remapping because they stay within their block scope.
