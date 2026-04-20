{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module HHLO.IR.Builder
    ( Builder
    , runBuilder
    , Tensor(..)
    , emitOp
    , arg
    , argNamed
    , moduleFromBuilder
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
    { bsNextId   :: !Int
    , bsOps      :: ![Operation]
    , bsArgCount :: !Int
    }

-- | Monad for constructing a sequence of MLIR operations.
newtype Builder a = Builder (State BuildState a)
    deriving (Functor, Applicative, Monad, MonadState BuildState)

-- | A phantom-typed tensor reference used by the EDSL.
newtype Tensor (s :: Shape) (d :: DType) = Tensor
    { tensorValue :: ValueId
    }
    deriving (Eq, Show)

-- | Construct a 'TensorType' from type-level shape and dtype proxies.
tensorType :: forall s d. (KnownShape s, KnownDType d) => Proxy s -> Proxy d -> TensorType
tensorType _ _ = TensorType (shapeVal (Proxy @s)) (dtypeVal (Proxy @d))

-- | Run a 'Builder' action and produce a 'Function'.
-- Argument 'ValueId's are negative: @-1@ maps to @%arg0@, @-2@ to @%arg1@, etc.
-- Operation result IDs start at 0.
runBuilder :: forall s d. (KnownShape s, KnownDType d) => Text -> [FuncArg] -> Builder (Tensor s d) -> Function
runBuilder name args' builderAction =
    let Builder m = builderAction
        initState = BuildState 0 [] 0
        (Tensor finalVid, finalState) = runState m initState
        resultType = tensorType (Proxy @s) (Proxy @d)
    in Function name args' resultType (reverse $ bsOps finalState)

-- | Create a top-level 'Module' from a single function produced by a builder.
-- Argument names are automatically set to @arg0@, @arg1@, etc. so that
-- the signature and body SSA references line up.
moduleFromBuilder :: forall s d. (KnownShape s, KnownDType d) => Text -> [FuncArg] -> Builder (Tensor s d) -> Module
moduleFromBuilder name args' action =
    let renamed = zipWith (\i (FuncArg _ t) -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] args'
    in Module [runBuilder name renamed action]

-- | Emit a generic operation into the builder.
-- The caller must provide the operand types so that the pretty-printer
-- can emit the full function type @(operandTypes) -> resultType@.
emitOp :: Text -> [ValueId] -> [TensorType] -> [Attribute] -> TensorType -> Builder ValueId
emitOp name operands operandTypes attrs resultType = do
    n <- gets bsNextId
    let vid = ValueId n
    modify $ \s -> s
        { bsNextId = n + 1
        , bsOps = Operation name operands operandTypes attrs vid resultType : bsOps s
        }
    return vid

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
instance KnownDType 'Bool where dtypeVal _ = Bool
