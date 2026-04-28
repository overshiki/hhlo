{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}

module HHLO.IR.Builder
    ( Builder
    , runBuilder
    , runBuilder2
    , runBuilder3
    , runBuilderT
    , BuildState(..)
    , Tensor(..)
    , Tuple2(..)
    , Tuple3(..)
    , Tuple4(..)
    , Tuple5(..)
    , Tuple6(..)
    , Tuple7(..)
    , Tuple8(..)
    , Tuple(..)
    , TupleBuilder(..)
    , emitOp
    , emitOpN
    , emitOpRegions
    , emitOpRegionsN
    , emitReduce
    , emitReturn
    , runBlockBuilder
    , arg
    , argNamed
    , moduleFromBuilder
    , moduleFromBuilder2
    , moduleFromBuilder3
    , moduleFromBuilder4
    , moduleFromBuilder5
    , moduleFromBuilder6
    , moduleFromBuilder7
    , moduleFromBuilder8
    , moduleFromBuilderT
    , tensorType
    , KnownDType(..)
    ) where

import Control.Monad.State
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import GHC.TypeLits
import HHLO.Core.Types
import HHLO.IR.AST

-- | Mutable state accumulated while building a function.
data BuildState = BuildState
    { bsNextId       :: !Int
    , bsOps          :: ![Operation]
    , bsArgCount     :: !Int
    , bsBlockArgBase :: !Int
    }

-- | Monad for constructing a sequence of MLIR operations.
newtype Builder a = Builder (State BuildState a)
    deriving (Functor, Applicative, Monad, MonadState BuildState)

-- | A phantom-typed tensor reference used by the EDSL.
newtype Tensor (s :: Shape) (d :: DType) = Tensor
    { tensorValue :: ValueId
    }
    deriving (Eq, Show)

-- | A pair of tensors for functions with two results.
data Tuple2 (s1 :: Shape) (d1 :: DType) (s2 :: Shape) (d2 :: DType)
    = Tuple2 (Tensor s1 d1) (Tensor s2 d2)
    deriving (Eq, Show)

-- | Three heterogeneous tensors.
data Tuple3 (s1 :: Shape) (d1 :: DType) (s2 :: Shape) (d2 :: DType) (s3 :: Shape) (d3 :: DType)
    = Tuple3 (Tensor s1 d1) (Tensor s2 d2) (Tensor s3 d3)
    deriving (Eq, Show)

-- | Four heterogeneous tensors.
data Tuple4 (s1 :: Shape) (d1 :: DType) (s2 :: Shape) (d2 :: DType) (s3 :: Shape) (d3 :: DType) (s4 :: Shape) (d4 :: DType)
    = Tuple4 (Tensor s1 d1) (Tensor s2 d2) (Tensor s3 d3) (Tensor s4 d4)
    deriving (Eq, Show)

-- | Five heterogeneous tensors.
data Tuple5 (s1 :: Shape) (d1 :: DType) (s2 :: Shape) (d2 :: DType) (s3 :: Shape) (d3 :: DType) (s4 :: Shape) (d4 :: DType) (s5 :: Shape) (d5 :: DType)
    = Tuple5 (Tensor s1 d1) (Tensor s2 d2) (Tensor s3 d3) (Tensor s4 d4) (Tensor s5 d5)
    deriving (Eq, Show)

-- | Six heterogeneous tensors.
data Tuple6 (s1 :: Shape) (d1 :: DType) (s2 :: Shape) (d2 :: DType) (s3 :: Shape) (d3 :: DType) (s4 :: Shape) (d4 :: DType) (s5 :: Shape) (d5 :: DType) (s6 :: Shape) (d6 :: DType)
    = Tuple6 (Tensor s1 d1) (Tensor s2 d2) (Tensor s3 d3) (Tensor s4 d4) (Tensor s5 d5) (Tensor s6 d6)
    deriving (Eq, Show)

-- | Seven heterogeneous tensors.
data Tuple7 (s1 :: Shape) (d1 :: DType) (s2 :: Shape) (d2 :: DType) (s3 :: Shape) (d3 :: DType) (s4 :: Shape) (d4 :: DType) (s5 :: Shape) (d5 :: DType) (s6 :: Shape) (d6 :: DType) (s7 :: Shape) (d7 :: DType)
    = Tuple7 (Tensor s1 d1) (Tensor s2 d2) (Tensor s3 d3) (Tensor s4 d4) (Tensor s5 d5) (Tensor s6 d6) (Tensor s7 d7)
    deriving (Eq, Show)

-- | Eight heterogeneous tensors.
data Tuple8 (s1 :: Shape) (d1 :: DType) (s2 :: Shape) (d2 :: DType) (s3 :: Shape) (d3 :: DType) (s4 :: Shape) (d4 :: DType) (s5 :: Shape) (d5 :: DType) (s6 :: Shape) (d6 :: DType) (s7 :: Shape) (d7 :: DType) (s8 :: Shape) (d8 :: DType)
    = Tuple8 (Tensor s1 d1) (Tensor s2 d2) (Tensor s3 d3) (Tensor s4 d4) (Tensor s5 d5) (Tensor s6 d6) (Tensor s7 d7) (Tensor s8 d8)
    deriving (Eq, Show)

-- | A heterogeneous tuple of tensors for multi-result functions.
data Tuple (ss :: [Shape]) (ds :: [DType]) where
    TNil  :: Tuple '[] '[]
    (:::) :: Tensor s d -> Tuple ss ds -> Tuple (s ': ss) (d ': ds)

infixr 5 :::

-- | Type class for extracting 'ValueId's and 'TensorType's from a 'Tuple'.
class TupleBuilder t where
    tupleVids   :: t -> [ValueId]
    tupleTypes  :: Proxy t -> [TensorType]

instance TupleBuilder (Tuple '[] '[]) where
    tupleVids TNil  = []
    tupleTypes _    = []

instance (KnownShape s, KnownDType d, TupleBuilder (Tuple ss ds))
      => TupleBuilder (Tuple (s ': ss) (d ': ds)) where
    tupleVids (t ::: ts) = tensorValue t : tupleVids ts
    tupleTypes _ = tensorType (Proxy @s) (Proxy @d) : tupleTypes (Proxy @(Tuple ss ds))

-- | Construct a 'TensorType' from type-level shape and dtype proxies.
tensorType :: forall s d. (KnownShape s, KnownDType d) => Proxy s -> Proxy d -> TensorType
tensorType _ _ = TensorType (shapeVal (Proxy @s)) (dtypeVal (Proxy @d))

-- | Run a 'Builder' action and produce a single-result 'Function'.
-- Argument 'ValueId's are negative: @-1@ maps to @%arg0@, @-2@ to @%arg1@, etc.
-- Operation result IDs start at 0.
runBuilder :: forall s d. (KnownShape s, KnownDType d) => Text -> [FuncArg] -> Builder (Tensor s d) -> Function
runBuilder name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tensor finalVid, finalState) = runState m initState
        resultType = tensorType (Proxy @s) (Proxy @d)
        ops = reverse $ bsOps finalState
    in Function name args' [resultType] [finalVid] ops

-- | Run a 'Builder' action and produce a two-result 'Function'.
runBuilder2 :: forall s1 d1 s2 d2. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
            => Text -> [FuncArg] -> Builder (Tuple2 s1 d1 s2 d2) -> Function
runBuilder2 name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tuple2 (Tensor v1) (Tensor v2), finalState) = runState m initState
        rt1 = tensorType (Proxy @s1) (Proxy @d1)
        rt2 = tensorType (Proxy @s2) (Proxy @d2)
        ops = reverse $ bsOps finalState
    in Function name args' [rt1, rt2] [v1, v2] ops

runBuilder3 :: forall s1 d1 s2 d2 s3 d3. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3)
            => Text -> [FuncArg] -> Builder (Tuple3 s1 d1 s2 d2 s3 d3) -> Function
runBuilder3 name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tuple3 (Tensor v1) (Tensor v2) (Tensor v3), finalState) = runState m initState
        rt1 = tensorType (Proxy @s1) (Proxy @d1)
        rt2 = tensorType (Proxy @s2) (Proxy @d2)
        rt3 = tensorType (Proxy @s3) (Proxy @d3)
        ops = reverse $ bsOps finalState
    in Function name args' [rt1, rt2, rt3] [v1, v2, v3] ops

runBuilder4 :: forall s1 d1 s2 d2 s3 d3 s4 d4. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4)
            => Text -> [FuncArg] -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4) -> Function
runBuilder4 name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tuple4 (Tensor v1) (Tensor v2) (Tensor v3) (Tensor v4), finalState) = runState m initState
        rt1 = tensorType (Proxy @s1) (Proxy @d1)
        rt2 = tensorType (Proxy @s2) (Proxy @d2)
        rt3 = tensorType (Proxy @s3) (Proxy @d3)
        rt4 = tensorType (Proxy @s4) (Proxy @d4)
        ops = reverse $ bsOps finalState
    in Function name args' [rt1, rt2, rt3, rt4] [v1, v2, v3, v4] ops

runBuilder5 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5)
            => Text -> [FuncArg] -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5) -> Function
runBuilder5 name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tuple5 (Tensor v1) (Tensor v2) (Tensor v3) (Tensor v4) (Tensor v5), finalState) = runState m initState
        rt1 = tensorType (Proxy @s1) (Proxy @d1)
        rt2 = tensorType (Proxy @s2) (Proxy @d2)
        rt3 = tensorType (Proxy @s3) (Proxy @d3)
        rt4 = tensorType (Proxy @s4) (Proxy @d4)
        rt5 = tensorType (Proxy @s5) (Proxy @d5)
        ops = reverse $ bsOps finalState
    in Function name args' [rt1, rt2, rt3, rt4, rt5] [v1, v2, v3, v4, v5] ops

runBuilder6 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6)
            => Text -> [FuncArg] -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6) -> Function
runBuilder6 name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tuple6 (Tensor v1) (Tensor v2) (Tensor v3) (Tensor v4) (Tensor v5) (Tensor v6), finalState) = runState m initState
        rt1 = tensorType (Proxy @s1) (Proxy @d1)
        rt2 = tensorType (Proxy @s2) (Proxy @d2)
        rt3 = tensorType (Proxy @s3) (Proxy @d3)
        rt4 = tensorType (Proxy @s4) (Proxy @d4)
        rt5 = tensorType (Proxy @s5) (Proxy @d5)
        rt6 = tensorType (Proxy @s6) (Proxy @d6)
        ops = reverse $ bsOps finalState
    in Function name args' [rt1, rt2, rt3, rt4, rt5, rt6] [v1, v2, v3, v4, v5, v6] ops

runBuilder7 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7)
            => Text -> [FuncArg] -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7) -> Function
runBuilder7 name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tuple7 (Tensor v1) (Tensor v2) (Tensor v3) (Tensor v4) (Tensor v5) (Tensor v6) (Tensor v7), finalState) = runState m initState
        rt1 = tensorType (Proxy @s1) (Proxy @d1)
        rt2 = tensorType (Proxy @s2) (Proxy @d2)
        rt3 = tensorType (Proxy @s3) (Proxy @d3)
        rt4 = tensorType (Proxy @s4) (Proxy @d4)
        rt5 = tensorType (Proxy @s5) (Proxy @d5)
        rt6 = tensorType (Proxy @s6) (Proxy @d6)
        rt7 = tensorType (Proxy @s7) (Proxy @d7)
        ops = reverse $ bsOps finalState
    in Function name args' [rt1, rt2, rt3, rt4, rt5, rt6, rt7] [v1, v2, v3, v4, v5, v6, v7] ops

runBuilder8 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7, KnownShape s8, KnownDType d8)
            => Text -> [FuncArg] -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8) -> Function
runBuilder8 name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (Tuple8 (Tensor v1) (Tensor v2) (Tensor v3) (Tensor v4) (Tensor v5) (Tensor v6) (Tensor v7) (Tensor v8), finalState) = runState m initState
        rt1 = tensorType (Proxy @s1) (Proxy @d1)
        rt2 = tensorType (Proxy @s2) (Proxy @d2)
        rt3 = tensorType (Proxy @s3) (Proxy @d3)
        rt4 = tensorType (Proxy @s4) (Proxy @d4)
        rt5 = tensorType (Proxy @s5) (Proxy @d5)
        rt6 = tensorType (Proxy @s6) (Proxy @d6)
        rt7 = tensorType (Proxy @s7) (Proxy @d7)
        rt8 = tensorType (Proxy @s8) (Proxy @d8)
        ops = reverse $ bsOps finalState
    in Function name args' [rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8] [v1, v2, v3, v4, v5, v6, v7, v8] ops

-- | Create a top-level 'Module' from a single-result function produced by a builder.
-- Argument names are automatically set to @arg0@, @arg1@, etc. so that
-- the signature and body SSA references line up.
moduleFromBuilder :: forall s d. (KnownShape s, KnownDType d) => Text -> [FuncArg] -> Builder (Tensor s d) -> Module
moduleFromBuilder name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder name renamed action]

-- | Create a top-level 'Module' from a two-result function produced by a builder.
moduleFromBuilder2 :: forall s1 d1 s2 d2. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2)
                   => Text -> [FuncArg] -> Builder (Tuple2 s1 d1 s2 d2) -> Module
moduleFromBuilder2 name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder2 name renamed action]

moduleFromBuilder3 :: forall s1 d1 s2 d2 s3 d3. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3)
                   => Text -> [FuncArg] -> Builder (Tuple3 s1 d1 s2 d2 s3 d3) -> Module
moduleFromBuilder3 name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder3 name renamed action]

moduleFromBuilder4 :: forall s1 d1 s2 d2 s3 d3 s4 d4. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4)
                   => Text -> [FuncArg] -> Builder (Tuple4 s1 d1 s2 d2 s3 d3 s4 d4) -> Module
moduleFromBuilder4 name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder4 name renamed action]

moduleFromBuilder5 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5)
                   => Text -> [FuncArg] -> Builder (Tuple5 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5) -> Module
moduleFromBuilder5 name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder5 name renamed action]

moduleFromBuilder6 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6)
                   => Text -> [FuncArg] -> Builder (Tuple6 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6) -> Module
moduleFromBuilder6 name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder6 name renamed action]

moduleFromBuilder7 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7)
                   => Text -> [FuncArg] -> Builder (Tuple7 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7) -> Module
moduleFromBuilder7 name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder7 name renamed action]

moduleFromBuilder8 :: forall s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8. (KnownShape s1, KnownDType d1, KnownShape s2, KnownDType d2, KnownShape s3, KnownDType d3, KnownShape s4, KnownDType d4, KnownShape s5, KnownDType d5, KnownShape s6, KnownDType d6, KnownShape s7, KnownDType d7, KnownShape s8, KnownDType d8)
                   => Text -> [FuncArg] -> Builder (Tuple8 s1 d1 s2 d2 s3 d3 s4 d4 s5 d5 s6 d6 s7 d7 s8 d8) -> Module
moduleFromBuilder8 name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder8 name renamed action]

-- | Run a 'Builder' action and produce a multi-result 'Function' from a 'Tuple'.
runBuilderT :: forall ss ds. TupleBuilder (Tuple ss ds) => Text -> [FuncArg] -> Builder (Tuple ss ds) -> Function
runBuilderT name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0 1000
        (tupleResult, finalState) = runState m initState
        rtypes = tupleTypes (Proxy @(Tuple ss ds))
        rvids  = tupleVids tupleResult
        ops    = reverse $ bsOps finalState
    in Function name args' rtypes rvids ops

-- | Create a top-level 'Module' from a multi-result function produced by a builder.
moduleFromBuilderT :: forall ss ds. TupleBuilder (Tuple ss ds) => Text -> [FuncArg] -> Builder (Tuple ss ds) -> Module
moduleFromBuilderT name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilderT name renamed action]

-- | Emit a generic single-result operation into the builder.
-- The caller must provide the operand types so that the pretty-printer
-- can emit the full function type @(operandTypes) -> resultType@.
emitOp :: Text -> [ValueId] -> [TensorType] -> [Attribute] -> TensorType -> Builder ValueId
emitOp name operands operandTypes attrs resultType =
    emitOpN name operands operandTypes attrs [resultType] >>= \case
        [vid] -> return vid
        _     -> error "emitOp: expected exactly one result"

-- | Emit a single-result operation that carries nested regions.
emitOpRegions :: Text -> [ValueId] -> [TensorType] -> [Attribute] -> [Region] -> TensorType -> Builder ValueId
emitOpRegions name operands operandTypes attrs regions resultType =
    emitOpRegionsN name operands operandTypes attrs regions [resultType] >>= \case
        [vid] -> return vid
        _     -> error "emitOpRegions: expected exactly one result"

-- | Emit a multi-result operation into the builder.
-- Returns a list of fresh 'ValueId's, one per result type.
emitOpN :: Text -> [ValueId] -> [TensorType] -> [Attribute] -> [TensorType] -> Builder [ValueId]
emitOpN name operands operandTypes attrs resultTypes =
    emitOpRegionsN name operands operandTypes attrs [] resultTypes

-- | Emit a multi-result operation that carries nested regions.
emitOpRegionsN :: Text -> [ValueId] -> [TensorType] -> [Attribute] -> [Region] -> [TensorType] -> Builder [ValueId]
emitOpRegionsN name operands operandTypes attrs regions resultTypes = do
    n <- gets bsNextId
    let numResults = length resultTypes
        vids = map ValueId [n .. n + numResults - 1]
    modify $ \s -> s
        { bsNextId = n + numResults
        , bsOps = Operation name operands operandTypes attrs regions vids resultTypes : bsOps s
        }
    return vids

-- | Run a nested builder action to produce a single 'Block'.
--
-- * Shares 'bsNextId' with the parent so SSA value IDs remain globally unique.
-- * Isolates 'bsOps' so inner ops do not leak into the parent's op list.
-- * Resets 'bsArgCount' so 'arg' inside the block creates fresh negative IDs.
runBlockBuilder :: [TensorType] -> Builder a -> Builder Block
runBlockBuilder argTypes (Builder inner) = do
    parent <- get
    let base = bsBlockArgBase parent
        blockArgs = zipWith (\i t -> FuncArg (T.pack ("arg" ++ show i)) t) [base..] argTypes
        innerState0 = BuildState (bsNextId parent) [] base (base + length argTypes)
        (_, innerState) = runState inner innerState0
    put $ parent { bsNextId = bsNextId innerState, bsBlockArgBase = bsBlockArgBase innerState }
    return $ Block blockArgs (reverse $ bsOps innerState)

-- | Emit a 'stablehlo.return' terminator inside a region.
--
-- The result type is a dummy scalar because 'return' has no operation results,
-- but the pretty-printer special-cases this op anyway.
emitReturn :: [ValueId] -> [TensorType] -> Builder ()
emitReturn vids types = do
    _ <- emitOpN "stablehlo.return" vids types [] [TensorType [] F32]
    return ()

-- | Emit a 'stablehlo.reduce' operation using the @applies@ shorthand.
--
-- Example (reduce sum over all dimensions):
-- @
--   vid <- emitReduce inputVid inputType initVid initType [0,1] "stablehlo.add" resultType
-- @
--
-- Emits:
-- @
--   %n = stablehlo.reduce(%input init: %init) applies stablehlo.add
--        across dimensions = [0, 1] : (input_type, init_type) -> result_type
-- @
emitReduce :: ValueId -> TensorType -> ValueId -> TensorType -> [Int] -> Text -> TensorType -> Builder ValueId
emitReduce input inType init_ initType dims appliesOp resultType =
    emitOpN "stablehlo.reduce"
        [input, init_] [inType, initType]
        [ AttrIntList "dimensions" (map fromIntegral dims)
        , AttrString "applies" appliesOp
        ] [resultType] >>= \case
            [vid] -> return vid
            _     -> error "emitReduce: expected exactly one result"

-- | Declare a function argument with an auto-generated name.
-- The returned 'Tensor' carries a negative 'ValueId' so that the
-- pretty-printer emits @%arg0@, @%arg1@, etc.
arg :: forall s d. (KnownShape s, KnownDType d) => Builder (Tensor s d)
arg = do
    n <- gets bsArgCount
    modify $ \s -> s { bsArgCount = n + 1 }
    let ttype = tensorType (Proxy @s) (Proxy @d)
    -- Negative IDs: -1 -> %arg0, -2 -> %arg1, ...
    return $ Tensor (ValueId (-(n + 1)))

-- | Declare a function argument with a specific name (for pretty-printing only).
argNamed :: forall s d. (KnownShape s, KnownDType d) => Text -> Builder (Tensor s d)
argNamed _name = arg @s @d

-- | Obtain the runtime value of a type-level 'DType'.
class KnownDType (d :: DType) where
    dtypeVal :: Proxy d -> DType

instance KnownDType 'F32  where dtypeVal _ = F32
instance KnownDType 'F64  where dtypeVal _ = F64
instance KnownDType 'I32  where dtypeVal _ = I32
instance KnownDType 'I64  where dtypeVal _ = I64
instance KnownDType 'I8   where dtypeVal _ = I8
instance KnownDType 'I16  where dtypeVal _ = I16
instance KnownDType 'UI8  where dtypeVal _ = UI8
instance KnownDType 'UI16 where dtypeVal _ = UI16
instance KnownDType 'UI32 where dtypeVal _ = UI32
instance KnownDType 'UI64 where dtypeVal _ = UI64
instance KnownDType 'Bool where dtypeVal _ = Bool
