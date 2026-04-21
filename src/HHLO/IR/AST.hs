{-# LANGUAGE StrictData #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.IR.AST
    ( ValueId(..)
    , valueRef
    , TensorType(..)
    , Attribute(..)
    , Operation(..)
    , Block(..)
    , Region(..)
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

-- | Render a 'ValueId' as an MLIR value reference.
-- Non-negative IDs are operation results (@"%42"@).
-- Negative IDs are function arguments: @-1@ prints as @"%arg0"@,
-- @-2@ as @"%arg1"@, etc.
valueRef :: ValueId -> Text
valueRef (ValueId n)
    | n >= 0    = "%" <> T.pack (show n)
    | otherwise = "%arg" <> T.pack (show (abs n - 1))

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
    | AttrRaw     Text           -- ^ Printed verbatim (no quotes).  Needed for
                                 --   dialect attributes like @#stablehlo.gather<...>@.
    deriving (Eq, Show)

-- | A block inside a region (e.g. the reducer body of 'stablehlo.reduce').
data Block = Block
    { blockArgs       :: ![FuncArg]
    , blockOps        :: ![Operation]
    }
    deriving (Eq, Show)

-- | A region attached to an operation.
newtype Region = Region { unRegion :: [Block] }
    deriving (Eq, Show)

-- | A single StableHLO operation.
data Operation = Operation
    { opName         :: !Text
    , opOperands     :: ![ValueId]
    , opOperandTypes :: ![TensorType]
    , opAttributes   :: ![Attribute]
    , opRegions      :: ![Region]
    , opResult       :: !ValueId
    , opResultType   :: !TensorType
    }
    deriving (Eq, Show)

-- | A function argument declaration.
data FuncArg = FuncArg
    { argName :: !Text
    , argType :: !TensorType
    }
    deriving (Eq, Show)

-- | A function definition inside a module.
-- Supports multiple result types for tuple-returning functions.
data Function = Function
    { funcName       :: !Text
    , funcArgs       :: ![FuncArg]
    , funcResults    :: ![TensorType]
    , funcReturnVids :: ![ValueId]
    , funcBody       :: ![Operation]
    }
    deriving (Eq, Show)

-- | A top-level MLIR module.
newtype Module = Module
    { moduleFunctions :: [Function]
    }
    deriving (Eq, Show)
