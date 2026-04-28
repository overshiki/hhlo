{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.Autograd.Grad
    ( grad
    , gradModule
    , grad2
    , gradModule2
    , grad3
    , gradModule3
    , vjp
    , vjpModule
    , inlineFunction
    , inlineFunction2
    , inlineFunction3
    ) where

import Control.Monad.State (gets)
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

-- | Compute gradients of a scalar-valued function w.r.t. two inputs.
gradModule2 :: forall s1 d1 s2 d2.
               (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
            => (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor '[] d1))
            -> Module
gradModule2 f = Module [gradFunction]
  where
    inType1 = tensorType (Proxy @s1) (Proxy @d1)
    inType2 = tensorType (Proxy @s2) (Proxy @d2)
    arg0 = FuncArg "arg0" inType1
    arg1 = FuncArg "arg1" inType2

    forwardFunc :: Function
    forwardFunc = runBuilder @'[] @d1 "forward" [arg0, arg1] $ do
        input1 <- arg @s1 @d1
        input2 <- arg @s2 @d2
        f input1 input2

    forwardOps :: [Operation]
    forwardOps = funcBody forwardFunc

    gradFunction :: Function
    gradFunction = runBuilder2 @s1 @d1 @s2 @d2 "main" [arg0, arg1] $ do
        input1 <- arg @s1 @d1
        input2 <- arg @s2 @d2
        y      <- f input1 input2

        seed <- constant @'[] @d1 1.0
        let initMap = Map.singleton (tensorValue y) (bfromTyped seed)
        finalMap <- foldlBackward backwardStep initMap (reverse forwardOps)

        let g1 = case Map.lookup (tensorValue input1) finalMap of
                    Just bt -> btoTyped @s1 bt
                    Nothing -> error "autograd-hhlo: gradient not found for input1"
            g2 = case Map.lookup (tensorValue input2) finalMap of
                    Just bt -> btoTyped @s2 bt
                    Nothing -> error "autograd-hhlo: gradient not found for input2"
        return (Tuple2 g1 g2)

-- | Compute gradients of a scalar-valued function w.r.t. three inputs.
gradModule3 :: forall s1 d1 s2 d2 s3 d3.
               ( KnownShape s1, KnownDType d1
               , KnownShape s2, KnownDType d2
               , KnownShape s3, KnownDType d3
               )
            => (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tensor '[] d1))
            -> Module
gradModule3 f = Module [gradFunction]
  where
    inType1 = tensorType (Proxy @s1) (Proxy @d1)
    inType2 = tensorType (Proxy @s2) (Proxy @d2)
    inType3 = tensorType (Proxy @s3) (Proxy @d3)
    arg0 = FuncArg "arg0" inType1
    arg1 = FuncArg "arg1" inType2
    arg2 = FuncArg "arg2" inType3

    forwardFunc :: Function
    forwardFunc = runBuilder @'[] @d1 "forward" [arg0, arg1, arg2] $ do
        input1 <- arg @s1 @d1
        input2 <- arg @s2 @d2
        input3 <- arg @s3 @d3
        f input1 input2 input3

    forwardOps :: [Operation]
    forwardOps = funcBody forwardFunc

    gradFunction :: Function
    gradFunction = runBuilder3 @s1 @d1 @s2 @d2 @s3 @d3 "main" [arg0, arg1, arg2] $ do
        input1 <- arg @s1 @d1
        input2 <- arg @s2 @d2
        input3 <- arg @s3 @d3
        y      <- f input1 input2 input3

        seed <- constant @'[] @d1 1.0
        let initMap = Map.singleton (tensorValue y) (bfromTyped seed)
        finalMap <- foldlBackward backwardStep initMap (reverse forwardOps)

        let g1 = case Map.lookup (tensorValue input1) finalMap of
                    Just bt -> btoTyped @s1 bt
                    Nothing -> error "autograd-hhlo: gradient not found for input1"
            g2 = case Map.lookup (tensorValue input2) finalMap of
                    Just bt -> btoTyped @s2 bt
                    Nothing -> error "autograd-hhlo: gradient not found for input2"
            g3 = case Map.lookup (tensorValue input3) finalMap of
                    Just bt -> btoTyped @s3 bt
                    Nothing -> error "autograd-hhlo: gradient not found for input3"
        return (Tuple3 g1 g2 g3)

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

-- | Two-argument gradient inside an existing 'Builder' context.
grad2 :: forall s1 d1 s2 d2.
         (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
      => (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor '[] d1))
      -> Tensor s1 d1 -> Tensor s2 d2
      -> Builder (Tensor s1 d1, Tensor s2 d2)
grad2 f input1 input2 = do
    let gradMod = gradModule2 f
        gradFunc = head (moduleFunctions gradMod)
    (g1, g2) <- inlineFunction2 gradFunc $ Map.fromList
        [ (ValueId (-1), tensorValue input1)
        , (ValueId (-2), tensorValue input2)
        ]
    return (Tensor g1, Tensor g2)

-- | Three-argument gradient inside an existing 'Builder' context.
grad3 :: forall s1 d1 s2 d2 s3 d3.
         ( KnownShape s1, KnownDType d1
         , KnownShape s2, KnownDType d2
         , KnownShape s3, KnownDType d3
         )
      => (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tensor '[] d1))
      -> Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3
      -> Builder (Tensor s1 d1, Tensor s2 d2, Tensor s3 d3)
grad3 f input1 input2 input3 = do
    let gradMod = gradModule3 f
        gradFunc = head (moduleFunctions gradMod)
    (g1, g2, g3) <- inlineFunction3 gradFunc $ Map.fromList
        [ (ValueId (-1), tensorValue input1)
        , (ValueId (-2), tensorValue input2)
        , (ValueId (-3), tensorValue input3)
        ]
    return (Tensor g1, Tensor g2, Tensor g3)

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
--
-- To avoid collisions between region-local IDs and the outer builder's
-- IDs, all positive IDs in the inlined function are shifted by the
-- current 'bsNextId' offset before inlining.
inlineFunction :: Function -> Map ValueId ValueId -> Builder ValueId
inlineFunction func argMap = do
    offset <- gets bsNextId
    let shiftedFunc = shiftFunctionIds offset func
        shiftedArgMap = Map.mapKeys (shiftVid offset) argMap
    finalMap <- foldlM inlineOp shiftedArgMap (funcBody shiftedFunc)
    let retId = head (funcReturnVids shiftedFunc)
    return $ Map.findWithDefault retId retId finalMap

-- | Inline a two-result function, returning both result value IDs.
inlineFunction2 :: Function -> Map ValueId ValueId -> Builder (ValueId, ValueId)
inlineFunction2 func argMap = do
    offset <- gets bsNextId
    let shiftedFunc = shiftFunctionIds offset func
        shiftedArgMap = Map.mapKeys (shiftVid offset) argMap
    finalMap <- foldlM inlineOp shiftedArgMap (funcBody shiftedFunc)
    let [retId1, retId2] = funcReturnVids shiftedFunc
    return ( Map.findWithDefault retId1 retId1 finalMap
           , Map.findWithDefault retId2 retId2 finalMap
           )

-- | Inline a three-result function, returning all three result value IDs.
inlineFunction3 :: Function -> Map ValueId ValueId -> Builder (ValueId, ValueId, ValueId)
inlineFunction3 func argMap = do
    offset <- gets bsNextId
    let shiftedFunc = shiftFunctionIds offset func
        shiftedArgMap = Map.mapKeys (shiftVid offset) argMap
    finalMap <- foldlM inlineOp shiftedArgMap (funcBody shiftedFunc)
    let [retId1, retId2, retId3] = funcReturnVids shiftedFunc
    return ( Map.findWithDefault retId1 retId1 finalMap
           , Map.findWithDefault retId2 retId2 finalMap
           , Map.findWithDefault retId3 retId3 finalMap
           )

-- | Shift all positive value IDs in a function by a given offset.
shiftFunctionIds :: Int -> Function -> Function
shiftFunctionIds offset func = func
    { funcBody = map (shiftOp offset) (funcBody func)
    , funcReturnVids = map (shiftVid offset) (funcReturnVids func)
    }

shiftOp :: Int -> Operation -> Operation
shiftOp offset op = op
    { opOperands = map (shiftVid offset) (opOperands op)
    , opResults  = map (shiftVid offset) (opResults op)
    , opRegions  = map (shiftRegion offset) (opRegions op)
    }

shiftRegion :: Int -> Region -> Region
shiftRegion offset (Region blocks) = Region (map (shiftBlock offset) blocks)

shiftBlock :: Int -> Block -> Block
shiftBlock offset (Block args ops) = Block args (map (shiftOp offset) ops)

shiftVid :: Int -> ValueId -> ValueId
shiftVid offset (ValueId n) = if n >= 0 then ValueId (n + offset) else ValueId n

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
