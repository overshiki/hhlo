{-# LANGUAGE StrictData #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.IR.AST
    ( ValueId(..)
    , valueRef
    , TensorType(..)
    , Attribute(..)
    , Operation(..)
    , FuncArg(..)
    , Function(..)
    , Module(..)
    ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Int (Int64)
import HHLO.Core.Types (DType, dtypeToText)

-- | An SSA value identifier in MLIR.
newtype ValueId = ValueId { unValueId :: Int }
    deriving (Eq, Ord, Show, Num)

-- | Render a 'ValueId' as an MLIR value reference (e.g. @"%42"@).
valueRef :: ValueId -> Text
valueRef (ValueId n)
    | n >= 0    = "%" <> T.pack (show n)
    | otherwise = "%neg" <> T.pack (show (abs n))

-- | A concrete tensor type with runtime-known shape.
data TensorType = TensorType
    { ttShape :: [Integer]   -- ^ Empty list denotes a scalar (@tensor<T>@).
    , ttDType :: DType
    }
    deriving (Eq, Show)

-- | Attributes attached to MLIR operations.
data Attribute
    = AttrInt     Text Int64
    | AttrFloat   Text Double
    | AttrBool    Text Bool
    | AttrString  Text Text
    | AttrIntList Text [Int64]
    | AttrDenseElements [Integer] DType [Double]
    | AttrDict    [(Text, Attribute)]
    deriving (Eq, Show)

-- | A single StableHLO operation.
data Operation = Operation
    { opName       :: !Text
    , opOperands   :: ![ValueId]
    , opAttributes :: ![Attribute]
    , opResult     :: !ValueId
    , opResultType :: !TensorType
    }
    deriving (Eq, Show)

-- | A function argument declaration.
data FuncArg = FuncArg
    { argName :: !Text
    , argType :: !TensorType
    }
    deriving (Eq, Show)

-- | A function definition inside a module.
data Function = Function
    { funcName   :: !Text
    , funcArgs   :: ![FuncArg]
    , funcResult :: !TensorType
    , funcBody   :: ![Operation]
    }
    deriving (Eq, Show)

-- | A top-level MLIR module.
newtype Module = Module
    { moduleFunctions :: [Function]
    }
    deriving (Eq, Show)
