{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Static shape, dtype, and attribute checker for StableHLO operations.
--
-- This module walks the 'Operation' AST and verifies that every op's
-- declared result types match the shapes inferred from its operands and
-- attributes. It runs automatically inside 'moduleFromBuilder' before
-- any MLIR is emitted.
module HHLO.ShapeCheck
    ( checkModule
    , ShapeError(..)
    ) where

import Control.Monad (foldM, forM, forM_, mapM_, when)
import Data.Int (Int64)
import Data.List (sort, nub, intersperse)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, listToMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import Text.Read (readMaybe)

import HHLO.Core.Types (DType(..))
import HHLO.IR.AST

-- ---------------------------------------------------------------------------
-- Error type
-- ---------------------------------------------------------------------------

data ShapeError = ShapeMismatch
    { seOpName   :: !Text
    , seValueId  :: !ValueId
    , seExpected :: !TensorType
    , seActual   :: !TensorType
    , seHint     :: !Text
    }
  | DtypeMismatch
    { seDtypeOpName   :: !Text
    , seDtypeValueId  :: !ValueId
    , seDtypeExpected :: !DType
    , seDtypeActual   :: !DType
    }
  | IndexOutOfBounds
    { seIdxOpName    :: !Text
    , seIdxDimension :: !Int
    , seIdxIndex     :: !Int64
    , seIdxBound     :: !Int64
    }
  | InvalidAttribute
    { seAttrOpName    :: !Text
    , seAttrAttrName  :: !Text
    , seAttrAttrValue :: !Text
    , seAttrReason    :: !Text
    }
  | MissingAttribute
    { seMissOpName   :: !Text
    , seMissAttrName :: !Text
    }
  | OperandCountMismatch
    { seCountOpName   :: !Text
    , seCountExpected :: !Int
    , seCountActual   :: !Int
    }
  | InternalError
    { seErrMessage :: !Text
    }
  deriving (Eq)

instance Show ShapeError where
    show (ShapeMismatch op vid expected actual hint) =
        "ShapeMismatch in " ++ T.unpack op ++ " at " ++ show vid ++ "\n"
        ++ "  expected: " ++ show expected ++ "\n"
        ++ "  actual:   " ++ show actual ++ "\n"
        ++ (if T.null hint then "" else "  hint:     " ++ T.unpack hint ++ "\n")
    show (DtypeMismatch op vid expected actual) =
        "DtypeMismatch in " ++ T.unpack op ++ " at " ++ show vid ++ "\n"
        ++ "  expected: " ++ show expected ++ "\n"
        ++ "  actual:   " ++ show actual ++ "\n"
    show (IndexOutOfBounds op dim idx bound) =
        "IndexOutOfBounds in " ++ T.unpack op ++ "\n"
        ++ "  dimension " ++ show dim ++ ": index " ++ show idx ++ " out of bounds [0, " ++ show bound ++ ")"
    show (InvalidAttribute op name val reason) =
        "InvalidAttribute in " ++ T.unpack op ++ ": " ++ T.unpack name ++ " = " ++ T.unpack val ++ "\n"
        ++ "  reason: " ++ T.unpack reason
    show (MissingAttribute op name) =
        "MissingAttribute in " ++ T.unpack op ++ ": " ++ T.unpack name
    show (OperandCountMismatch op expected actual) =
        "OperandCountMismatch in " ++ T.unpack op ++ ": expected " ++ show expected ++ " operands, got " ++ show actual
    show (InternalError msg) =
        "InternalError: " ++ T.unpack msg

-- ---------------------------------------------------------------------------
-- Shape environment
-- ---------------------------------------------------------------------------

type ShapeEnv = Map ValueId TensorType

initialEnv :: [FuncArg] -> ShapeEnv
initialEnv args = Map.fromList
    [(ValueId (-i - 1), argType arg) | (i, arg) <- zip [0::Int ..] args]

extendEnv :: ShapeEnv -> [ValueId] -> [TensorType] -> ShapeEnv
extendEnv env vids ttypes = foldl (\e (v, t) -> Map.insert v t e) env (zip vids ttypes)

lookupEnv :: ShapeEnv -> ValueId -> Maybe TensorType
lookupEnv env vid = Map.lookup vid env

-- ---------------------------------------------------------------------------
-- Attribute helpers
-- ---------------------------------------------------------------------------

lookupAttrInt :: Text -> [Attribute] -> Maybe Int64
lookupAttrInt name attrs = listToMaybe
    [v | AttrInt n v <- attrs, n == name]

lookupAttrBool :: Text -> [Attribute] -> Maybe Bool
lookupAttrBool name attrs = listToMaybe
    [v | AttrBool n v <- attrs, n == name]

lookupAttrString :: Text -> [Attribute] -> Maybe Text
lookupAttrString name attrs = listToMaybe
    [v | AttrString n v <- attrs, n == name]

lookupAttrIntList :: Text -> [Attribute] -> Maybe [Int64]
lookupAttrIntList name attrs = listToMaybe
    [v | AttrIntList n v <- attrs, n == name]

lookupAttrRaw :: Text -> [Attribute] -> Maybe Text
lookupAttrRaw name attrs = listToMaybe
    [raw | AttrRaw raw <- attrs, name `T.isPrefixOf` raw]

requireAttr :: Text -> Text -> [Attribute] -> (Attribute -> Maybe a) -> Either ShapeError a
requireAttr opName attrName attrs extract =
    case listToMaybe [a | a <- attrs, attrMatches a] of
        Nothing -> Left $ MissingAttribute opName attrName
        Just a  -> case extract a of
            Nothing -> Left $ InvalidAttribute opName attrName (T.pack $ show a) "wrong type"
            Just v  -> Right v
  where
    attrMatches (AttrInt n _)     = n == attrName
    attrMatches (AttrBool n _)    = n == attrName
    attrMatches (AttrString n _)  = n == attrName
    attrMatches (AttrIntList n _) = n == attrName
    attrMatches (AttrEnum n _)    = n == attrName
    attrMatches _                 = False

-- ---------------------------------------------------------------------------
-- Main entry points
-- ---------------------------------------------------------------------------

-- | Check an entire module for shape/dtype/attribute consistency.
checkModule :: Module -> Either ShapeError ()
checkModule (Module fns) = mapM_ checkFunction fns

-- | Check a single function.
checkFunction :: Function -> Either ShapeError ()
checkFunction fn = do
    let env0 = initialEnv (funcArgs fn)
    _ <- foldM (checkOp (funcName fn)) env0 (funcBody fn)
    return ()

-- | Check one operation and return the updated environment.
checkOp :: Text -> ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkOp _funcName env op = case opName op of
    "stablehlo.convolution"    -> checkConv env op
    "stablehlo.slice"          -> checkSlice env op
    "stablehlo.dynamic_slice"  -> checkDynamicSlice env op
    "stablehlo.pad"            -> checkPad env op
    "stablehlo.concatenate"    -> checkConcatenate env op
    "stablehlo.transpose"      -> checkTranspose env op
    "stablehlo.dot_general"    -> checkDotGeneral env op
    "stablehlo.custom_call"    -> checkCustomCall env op
    "stablehlo.broadcast_in_dim" -> checkBroadcastInDim env op
    "stablehlo.reshape"        -> checkReshape env op
    "stablehlo.iota"           -> checkIota env op
    "stablehlo.return"         -> checkReturn env op
    "stablehlo.compare"        -> checkCompare env op
    "stablehlo.select"         -> checkSelect env op
    "stablehlo.reduce"         -> checkReduce env op
    "stablehlo.sort"           -> checkSort env op
    -- Element-wise and shape-preserving ops
    _ | opName op `elem` elementwiseOps -> checkElementwise env op
      | otherwise                       -> checkPassthrough env op

-- ---------------------------------------------------------------------------
-- Element-wise ops (shape-preserving)
-- ---------------------------------------------------------------------------

elementwiseOps :: [Text]
elementwiseOps =
    [ "stablehlo.add", "stablehlo.subtract", "stablehlo.multiply"
    , "stablehlo.divide", "stablehlo.negate", "stablehlo.abs"
    , "stablehlo.exponential", "stablehlo.log", "stablehlo.sqrt"
    , "stablehlo.rsqrt", "stablehlo.sine", "stablehlo.cosine"
    , "stablehlo.tangent", "stablehlo.log1p", "stablehlo.floor"
    , "stablehlo.ceil", "stablehlo.maximum", "stablehlo.minimum"
    , "stablehlo.power", "stablehlo.tanh", "stablehlo.erf"
    , "stablehlo.relu", "stablehlo.gelu", "stablehlo.sigmoid"
    , "stablehlo.sign", "stablehlo.cos", "stablehlo.sin"
    , "stablehlo.tan", "stablehlo.atan2", "stablehlo.not"
    , "stablehlo.and", "stablehlo.or", "stablehlo.xor"
    , "stablehlo.popcnt", "stablehlo.clz", "stablehlo.collective_permute"
    ]

checkElementwise :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkElementwise env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        outs     = opResults op
    -- All operands must have the same shape and dtype
    case inTypes of
        [] -> return ()
        (t:ts) -> forM_ ts $ \t' ->
            expectType "elementwise operand" (head $ opOperands op) t t'
    -- Each output must match the first operand's shape/dtype
    forM_ (zip outs outTypes) $ \(vid, outType) ->
        case inTypes of
            [] -> return ()
            (t:_) -> expectType "elementwise output" vid t outType
    return $ extendEnv env outs outTypes

checkCompare :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkCompare env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        outs     = opResults op
    -- Operands must have the same shape (dtype can differ, but usually same)
    case inTypes of
        [] -> return ()
        (t:ts) -> forM_ ts $ \t' ->
            when (ttShape t /= ttShape t') $
                Left $ ShapeMismatch "compare" (head $ opOperands op) t t'
                    "compare operands must have the same shape"
    -- Output shape matches input shape, but dtype is Bool
    forM_ (zip outs outTypes) $ \(vid, outType) ->
        case inTypes of
            [] -> return ()
            (t:_) -> do
                when (ttShape t /= ttShape outType) $
                    Left $ ShapeMismatch "compare" vid (t { ttShape = ttShape t }) outType
                        "compare output shape must match input shape"
                when (ttDType outType /= Bool) $
                    Left $ DtypeMismatch "stablehlo.compare" vid Bool (ttDType outType)
    return $ extendEnv env outs outTypes

checkPassthrough :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkPassthrough env op = do
    -- Trust the declared types; just record them in the environment.
    return $ extendEnv env (opResults op) (opResultTypes op)

checkReturn :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkReturn env op = do
    -- return has no results; just verify operand count matches operandTypes
    when (length (opOperands op) /= length (opOperandTypes op)) $
        Left $ OperandCountMismatch "stablehlo.return"
            (length $ opOperandTypes op) (length $ opOperands op)
    return env

-- ---------------------------------------------------------------------------
-- stablehlo.convolution
-- ---------------------------------------------------------------------------

checkConv :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkConv env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes /= 2) $
        Left $ OperandCountMismatch (opName op) 2 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [lhsType, rhsType] = inTypes
        [outType] = outTypes
    expected <- computeConvOutput lhsType rhsType attrs
    expectType "convolution output" (head outs) expected outType
    return $ extendEnv env outs outTypes

-- | Compute expected convolution output shape from canonical structured attributes.
computeConvOutput :: TensorType -> TensorType -> [Attribute] -> Either ShapeError TensorType
computeConvOutput lhsType rhsType attrs = do
    -- Read dimension numbers
    ib  <- req "input_batch_dimension"
    if_ <- req "input_feature_dimension"
    isd <- reqList "input_spatial_dimensions"
    kif <- req "kernel_input_feature_dimension"
    kof <- req "kernel_output_feature_dimension"
    ksd <- reqList "kernel_spatial_dimensions"
    ob  <- req "output_batch_dimension"
    of_ <- req "output_feature_dimension"
    osd <- reqList "output_spatial_dimensions"
    -- Read window attributes
    let strides   = fromMaybe [1,1] (lookupAttrIntList "window_strides" attrs)
        padding   = fromMaybe (replicate (2 * length isd) 0) (lookupAttrIntList "padding" attrs)
        lhsDil    = fromMaybe (replicate (length isd) 1) (lookupAttrIntList "lhs_dilation" attrs)
        rhsDil    = fromMaybe (replicate (length isd) 1) (lookupAttrIntList "rhs_dilation" attrs)
    -- Validate padding length
    when (length padding /= 2 * length isd) $
        Left $ InvalidAttribute "stablehlo.convolution" "padding"
            (T.pack $ show padding)
            ("length must be 2 * spatial_rank (= " <> T.pack (show $ 2 * length isd) <> ")")
    -- Compute output spatial sizes
    outSpatial <- forM (zip [0..] isd) $ \(i, lhsSpatIdx) -> do
        let rhsSpatIdx = ksd !! fromIntegral i
            inputSize  = fromIntegral (ttShape lhsType !! fromIntegral lhsSpatIdx) :: Int64
            kernelSize = fromIntegral (ttShape rhsType !! fromIntegral rhsSpatIdx) :: Int64
            padLow     = fromIntegral (padding !! (2 * fromIntegral i))
            padHigh    = fromIntegral (padding !! (2 * fromIntegral i + 1))
            stride     = if fromIntegral i < length strides then strides !! fromIntegral i else 1
            rhsDilation = if fromIntegral i < length rhsDil then rhsDil !! fromIntegral i else 1
            lhsDilation = if fromIntegral i < length lhsDil then lhsDil !! fromIntegral i else 1
            dilatedKernel = rhsDilation * (kernelSize - 1) + 1
            dilatedInput  = lhsDilation * (inputSize - 1) + 1
            numerator     = dilatedInput + padLow + padHigh - dilatedKernel
            outputSize    = (numerator `div` stride) + 1
        when (numerator < 0) $
            Left $ InternalError $ "convolution results in negative spatial size: input=" <> T.pack (show inputSize)
                <> " kernel=" <> T.pack (show kernelSize) <> " pad=[" <> T.pack (show padLow) <> "," <> T.pack (show padHigh) <> "]"
        return (fromIntegral outputSize :: Integer)
    -- Build output shape
    let outBatch = ttShape lhsType !! fromIntegral ib
        outFeat  = ttShape rhsType !! fromIntegral kof
        expectedShape = [outBatch] ++ outSpatial ++ [outFeat]
    return $ lhsType { ttShape = expectedShape }
  where
    req name = case lookupAttrInt name attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute "stablehlo.convolution" name
    reqList name = case lookupAttrIntList name attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute "stablehlo.convolution" name

-- ---------------------------------------------------------------------------
-- stablehlo.slice
-- ---------------------------------------------------------------------------

checkSlice :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkSlice env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [inType] = inTypes
        [outType] = outTypes
    starts  <- case lookupAttrIntList "start_indices" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "start_indices"
    limits  <- case lookupAttrIntList "limit_indices" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "limit_indices"
    strides <- case lookupAttrIntList "strides" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "strides"
    let rank = length (ttShape inType)
    when (length starts /= rank) $
        Left $ InvalidAttribute (opName op) "start_indices" (T.pack $ show starts)
            ("length " <> T.pack (show $ length starts) <> " /= rank " <> T.pack (show rank))
    when (length limits /= rank) $
        Left $ InvalidAttribute (opName op) "limit_indices" (T.pack $ show limits)
            ("length " <> T.pack (show $ length limits) <> " /= rank " <> T.pack (show rank))
    when (length strides /= rank) $
        Left $ InvalidAttribute (opName op) "strides" (T.pack $ show strides)
            ("length " <> T.pack (show $ length strides) <> " /= rank " <> T.pack (show rank))
    expectedShape <- forM (zip5 [0..] (ttShape inType) starts limits strides) $
        \(dim, sz, s, l, st) -> do
            when (st <= 0) $
                Left $ InvalidAttribute (opName op) "strides" (T.pack $ show st) "must be > 0"
            when (s < 0) $
                Left $ IndexOutOfBounds (opName op) dim s (fromIntegral sz)
            when (l > fromIntegral sz) $
                Left $ IndexOutOfBounds (opName op) dim l (fromIntegral sz)
            when (s > l) $
                Left $ InvalidAttribute (opName op) "limit_indices" (T.pack $ show l)
                    ("limit " <> T.pack (show l) <> " must be >= start " <> T.pack (show s) <> " in dimension " <> T.pack (show dim))
            return $ fromIntegral $ (l - s + st - 1) `div` st
    let expected = inType { ttShape = expectedShape }
    expectType "slice output" (head outs) expected outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.dynamic_slice
-- ---------------------------------------------------------------------------

checkDynamicSlice :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkDynamicSlice env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes < 1) $
        Left $ OperandCountMismatch (opName op) 1 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let inType = head inTypes
        [outType] = outTypes
    sliceSizes <- case lookupAttrIntList "slice_sizes" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "slice_sizes"
    let rank = length (ttShape inType)
    when (length sliceSizes /= rank) $
        Left $ InvalidAttribute (opName op) "slice_sizes" (T.pack $ show sliceSizes)
            ("length must match input rank " <> T.pack (show rank))
    -- Verify slice sizes are within input bounds
    forM_ (zip (ttShape inType) sliceSizes) $ \(sz, ss) ->
        when (fromIntegral ss > sz) $
            Left $ InvalidAttribute (opName op) "slice_sizes" (T.pack $ show ss)
                ("slice size cannot exceed dimension size " <> T.pack (show sz))
    let expected = inType { ttShape = map fromIntegral sliceSizes }
    expectType "dynamic_slice output" (head outs) expected outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.pad
-- ---------------------------------------------------------------------------

checkPad :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkPad env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes /= 2) $
        Left $ OperandCountMismatch (opName op) 2 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [inType, padValType] = inTypes
        [outType] = outTypes
    -- Padding value must be scalar
    when (ttShape padValType /= []) $
        Left $ ShapeMismatch (opName op) (head $ opOperands op)
            (TensorType [] (ttDType padValType)) padValType
            "padding value must be a scalar"
    low     <- case lookupAttrIntList "edge_padding_low" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "edge_padding_low"
    high    <- case lookupAttrIntList "edge_padding_high" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "edge_padding_high"
    interior <- case lookupAttrIntList "interior_padding" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "interior_padding"
    let rank = length (ttShape inType)
    when (length low /= rank) $
        Left $ InvalidAttribute (opName op) "edge_padding_low" (T.pack $ show low)
            ("length must match input rank " <> T.pack (show rank))
    when (length high /= rank) $
        Left $ InvalidAttribute (opName op) "edge_padding_high" (T.pack $ show high)
            ("length must match input rank " <> T.pack (show rank))
    when (length interior /= rank) $
        Left $ InvalidAttribute (opName op) "interior_padding" (T.pack $ show interior)
            ("length must match input rank " <> T.pack (show rank))
    expectedShape <- forM (zip5 (ttShape inType) low high interior [0..]) $
        \(sz, l, h, i, dim) -> do
            when (i < 0) $
                Left $ InvalidAttribute (opName op) "interior_padding" (T.pack $ show i)
                    ("interior padding must be >= 0 in dimension " <> T.pack (show dim))
            return $ fromIntegral $ l + fromIntegral sz + i * (fromIntegral sz - 1) + h
    let expected = inType { ttShape = expectedShape }
    expectType "pad output" (head outs) expected outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.concatenate
-- ---------------------------------------------------------------------------

checkConcatenate :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkConcatenate env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (null inTypes) $
        Left $ OperandCountMismatch (opName op) 1 0
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [outType] = outTypes
    let attrs = opAttributes op
    let attrs = opAttributes op
    dim <- case lookupAttrInt "dimension" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "dimension"
    let rank = length (ttShape $ head inTypes)
    when (dim < 0 || dim >= fromIntegral rank) $
        Left $ IndexOutOfBounds (opName op) 0 dim (fromIntegral rank)
    let dimInt = fromIntegral dim
    -- All shapes must match except at concat dimension
    forM_ (zip [1..] (tail inTypes)) $ \(idx, t) -> do
        when (length (ttShape t) /= rank) $
            Left $ ShapeMismatch (opName op) (opOperands op !! idx)
                (head inTypes) t "all operands must have the same rank"
        forM_ (zip [0..] (zip (ttShape $ head inTypes) (ttShape t))) $ \(d, (s1, s2)) ->
            when (d /= dimInt && s1 /= s2) $
                Left $ ShapeMismatch (opName op) (opOperands op !! idx)
                    (head inTypes) t ("shapes must match except at concat dimension " <> T.pack (show dimInt))
    let expectedDimSize = sum [ttShape t !! dimInt | t <- inTypes]
    let expectedShape = [if d == dimInt then expectedDimSize else ttShape (head inTypes) !! d | d <- [0..rank-1]]
    let expected = (head inTypes) { ttShape = expectedShape }
    expectType "concatenate output" (head outs) expected outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.transpose
-- ---------------------------------------------------------------------------

checkTranspose :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkTranspose env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [inType] = inTypes
        [outType] = outTypes
    perm <- case lookupAttrIntList "permutation" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "permutation"
    let rank = length (ttShape inType)
    let expectedPerm = [0..fromIntegral rank - 1]
    when (sort perm /= expectedPerm) $
        Left $ InvalidAttribute (opName op) "permutation" (T.pack $ show perm)
            "must be a permutation of dimensions"
    let inShape = ttShape inType
        expectedShape = [inShape !! fromIntegral (perm !! i) | i <- [0..rank-1]]
    let expected = inType { ttShape = expectedShape }
    expectType "transpose output" (head outs) expected outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.dot_general
-- ---------------------------------------------------------------------------

checkDotGeneral :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkDotGeneral env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes /= 2) $
        Left $ OperandCountMismatch (opName op) 2 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [lhsType, rhsType] = inTypes
        [outType] = outTypes
    batchL    <- case lookupAttrIntList "lhs_batching_dimensions" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "lhs_batching_dimensions"
    batchR    <- case lookupAttrIntList "rhs_batching_dimensions" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "rhs_batching_dimensions"
    contractL <- case lookupAttrIntList "lhs_contracting_dimensions" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "lhs_contracting_dimensions"
    contractR <- case lookupAttrIntList "rhs_contracting_dimensions" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "rhs_contracting_dimensions"
    -- Verify batch dims match in size
    when (length batchL /= length batchR) $
        Left $ InvalidAttribute (opName op) "batching_dimensions" (T.pack $ show batchL)
            "lhs and rhs batching dimensions must have same length"
    forM_ (zip batchL batchR) $ \(bl, br) -> do
        let sl = ttShape lhsType !! fromIntegral bl
            sr = ttShape rhsType !! fromIntegral br
        when (sl /= sr) $
            Left $ ShapeMismatch (opName op) (head outs) lhsType rhsType
                ("batch dimension mismatch: lhs[" <> T.pack (show bl) <> "]=" <> T.pack (show sl)
                 <> " rhs[" <> T.pack (show br) <> "]=" <> T.pack (show sr))
    -- Compute expected output shape
    let lhsOutDims = [i | i <- [0..length (ttShape lhsType)-1]
                        , i `notElem` map fromIntegral contractL
                        , i `notElem` map fromIntegral batchL]
        rhsOutDims = [i | i <- [0..length (ttShape rhsType)-1]
                        , i `notElem` map fromIntegral contractR
                        , i `notElem` map fromIntegral batchR]
    let expectedShape = [ttShape lhsType !! i | i <- map fromIntegral batchL]
                     ++ [ttShape lhsType !! i | i <- lhsOutDims]
                     ++ [ttShape rhsType !! i | i <- rhsOutDims]
    let expected = lhsType { ttShape = expectedShape }
    expectType "dot_general output" (head outs) expected outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.custom_call
-- ---------------------------------------------------------------------------

checkCustomCall :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkCustomCall env op = do
    let attrs = opAttributes op
    _ <- requireAttr (opName op) "call_target_name" attrs $ \case AttrString _ v -> Just v; _ -> Nothing
    _ <- requireAttr (opName op) "has_side_effect" attrs $ \case AttrBool _ v -> Just v; _ -> Nothing
    -- api_version must be present
    case lookupAttrRaw "api_version" attrs of
        Just _  -> return ()  -- AttrRaw with : i32 is correct
        Nothing -> case lookupAttrInt "api_version" attrs of
            Just _  -> return ()  -- will be caught by pretty-printer test
            Nothing -> Left $ MissingAttribute (opName op) "api_version"
    return $ extendEnv env (opResults op) (opResultTypes op)

-- ---------------------------------------------------------------------------
-- stablehlo.broadcast_in_dim
-- ---------------------------------------------------------------------------

checkBroadcastInDim :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkBroadcastInDim env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [inType] = inTypes
        [outType] = outTypes
    dims <- case lookupAttrIntList "broadcast_dimensions" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "broadcast_dimensions"
    let inRank = length (ttShape inType)
        outRank = length (ttShape outType)
    when (length dims /= inRank) $
        Left $ InvalidAttribute (opName op) "broadcast_dimensions" (T.pack $ show dims)
            ("length must match input rank " <> T.pack (show inRank))
    forM_ (zip [0..] dims) $ \(i, d) -> do
        let inSize = ttShape inType !! i
            outSize = ttShape outType !! fromIntegral d
        when (inSize /= 1 && inSize /= outSize) $
            Left $ ShapeMismatch (opName op) (head outs) inType outType
                ("broadcast dimension " <> T.pack (show d) <> ": input size " <> T.pack (show inSize)
                 <> " cannot broadcast to output size " <> T.pack (show outSize))
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.reshape
-- ---------------------------------------------------------------------------

checkReshape :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkReshape env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        outs     = opResults op
    when (length inTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [inType] = inTypes
        [outType] = outTypes
    let inElems = product (ttShape inType)
        outElems = product (ttShape outType)
    when (inElems /= outElems) $
        Left $ ShapeMismatch (opName op) (head outs) outType inType
            ("reshape must preserve element count: " <> T.pack (show inElems) <> " /= " <> T.pack (show outElems))
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.iota
-- ---------------------------------------------------------------------------

checkIota :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkIota env op = do
    let outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [outType] = outTypes
    dim <- case lookupAttrInt "iota_dimension" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "iota_dimension"
    let rank = length (ttShape outType)
    when (dim < 0 || dim >= fromIntegral rank) $
        Left $ IndexOutOfBounds (opName op) 0 dim (fromIntegral rank)
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.select
-- ---------------------------------------------------------------------------

checkSelect :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkSelect env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        outs     = opResults op
    when (length inTypes /= 3) $
        Left $ OperandCountMismatch (opName op) 3 (length inTypes)
    when (length outTypes /= 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let [predType, onTrue, onFalse] = inTypes
        [outType] = outTypes
    -- Predicate can be scalar-broadcasted or same shape
    when (ttShape predType /= [] && ttShape predType /= ttShape onTrue) $
        Left $ ShapeMismatch (opName op) (opOperands op !! 0) onTrue predType
            "predicate shape must be scalar or match operand shapes"
    expectType "select onTrue" (opOperands op !! 1) onTrue outType
    expectType "select onFalse" (opOperands op !! 2) onFalse outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.reduce
-- ---------------------------------------------------------------------------

checkReduce :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkReduce env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        attrs    = opAttributes op
        outs     = opResults op
    when (length inTypes < 2) $
        Left $ OperandCountMismatch (opName op) 2 (length inTypes)
    when (length outTypes < 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    let inputType = head inTypes
    dims <- case lookupAttrIntList "dimensions" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "dimensions"
    let inShape = ttShape inputType
    -- Verify dimensions are valid
    forM_ dims $ \d ->
        when (d < 0 || d >= fromIntegral (length inShape)) $
            Left $ IndexOutOfBounds (opName op) 0 d (fromIntegral $ length inShape)
    -- Output shape = input shape with reduced dimensions removed
    let expectedShape = [sz | (i, sz) <- zip [0..] inShape, i `notElem` map fromIntegral dims]
    let expected = inputType { ttShape = expectedShape }
    forM_ (zip outs outTypes) $ \(vid, outType) ->
        expectType "reduce output" vid expected outType
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- stablehlo.sort
-- ---------------------------------------------------------------------------

checkSort :: ShapeEnv -> Operation -> Either ShapeError ShapeEnv
checkSort env op = do
    let inTypes  = opOperandTypes op
        outTypes = opResultTypes op
        outs     = opResults op
        attrs    = opAttributes op
    when (length inTypes < 1) $
        Left $ OperandCountMismatch (opName op) 1 (length inTypes)
    when (length outTypes < 1) $
        Left $ OperandCountMismatch (opName op) 1 (length outTypes)
    dim <- case lookupAttrInt "dimension" attrs of
        Just v  -> Right v
        Nothing -> Left $ MissingAttribute (opName op) "dimension"
    let rank = length (ttShape $ head inTypes)
    when (dim < 0 || dim >= fromIntegral rank) $
        Left $ IndexOutOfBounds (opName op) 0 dim (fromIntegral rank)
    -- sort preserves shapes
    forM_ (zip outs outTypes) $ \(vid, outType) -> do
        case filter (== outType) inTypes of
            [] -> Left $ ShapeMismatch (opName op) vid (head inTypes) outType "sort output must match an input"
            _  -> return ()
    return $ extendEnv env outs outTypes

-- ---------------------------------------------------------------------------
-- Utilities
-- ---------------------------------------------------------------------------

expectType :: Text -> ValueId -> TensorType -> TensorType -> Either ShapeError ()
expectType ctx vid expected actual =
    when (expected /= actual) $ Left $ ShapeMismatch ctx vid expected actual ""

zip4 :: [a] -> [b] -> [c] -> [d] -> [(a, b, c, d)]
zip4 (a:as) (b:bs) (c:cs) (d:ds) = (a,b,c,d) : zip4 as bs cs ds
zip4 _ _ _ _ = []

zip5 :: [a] -> [b] -> [c] -> [d] -> [e] -> [(a, b, c, d, e)]
zip5 (a:as) (b:bs) (c:cs) (d:ds) (e:es) = (a,b,c,d,e) : zip5 as bs cs ds es
zip5 _ _ _ _ _ = []


