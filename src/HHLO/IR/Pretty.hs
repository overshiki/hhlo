{-# LANGUAGE OverloadedStrings #-}

module HHLO.IR.Pretty
    ( Pretty(..)
    , render
    , renderLazy
    ) where

import Data.Int (Int64)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Lazy as TL
import Data.Text.Lazy.Builder (Builder, fromText, toLazyText)
import HHLO.IR.AST
import HHLO.Core.Types (DType(..), dtypeToText)

class Pretty a where
    pretty :: a -> Builder

render :: Pretty a => a -> Text
render = TL.toStrict . renderLazy

renderLazy :: Pretty a => a -> TL.Text
renderLazy = toLazyText . pretty

instance Pretty Module where
    pretty (Module funcs) =
        "module {\n"
        <> mconcat (map ((<> "\n") . indentFunc . pretty) funcs)
        <> "}"
      where
        indentFunc b =
            let ls = TL.splitOn (TL.pack "\n") (toLazyText b)
                indented = map (\l -> if TL.null l then l else TL.pack "  " <> l) ls
            in fromText (TL.toStrict (TL.intercalate (TL.pack "\n") indented))

instance Pretty Function where
    pretty (Function name args results returnVids ops) =
        fromText "func.func @" <> fromText name <> "("
        <> mconcat (intersperse (fromText ", ") (map pretty args))
        <> ") -> " <> prettyResults results <> " {\n"
        <> mconcat (map ((<> "\n") . ("    " <>)) (map pretty ops))
        <> returnLine returnVids results
        <> "}"

prettyResults :: [TensorType] -> Builder
prettyResults []  = "()"
prettyResults [r] = pretty r
prettyResults rs  = "(" <> mconcat (intersperse (fromText ", ") (map pretty rs)) <> ")"

prettyTypesList :: [TensorType] -> Builder
prettyTypesList []  = ""
prettyTypesList rs  = mconcat (intersperse (fromText ", ") (map pretty rs))

returnLine :: [ValueId] -> [TensorType] -> Builder
returnLine []     results = "    return : " <> prettyTypesList results <> "\n"
returnLine vids   results =
    let refs = mconcat (intersperse (fromText ", ") (map valueRefBuilder vids))
    in "    return " <> refs <> " : " <> prettyTypesList results <> "\n"

instance Pretty FuncArg where
    pretty (FuncArg name t) =
        fromText "%" <> fromText name <> ": " <> pretty t

instance Pretty Operation where
    pretty (Operation "stablehlo.reduce" operands operandTypes attrs regions results resultTypes)
        | null regions =
            -- Special format for reduce with 'applies' shorthand (no region):
            --   %n = stablehlo.reduce(%input init: %init) applies stablehlo.add
            --        across dimensions = [0] : (input_type, init_type) -> result_type
            prettyResultVids results <> " = stablehlo.reduce("
            <> valueRefBuilder (operands !! 0) <> " init: " <> valueRefBuilder (operands !! 1) <> ")"
            <> prettyReduceAttrs attrs
            <> " : " <> prettyResultType operandTypes resultTypes
        | otherwise =
            -- Generic form when a region is present:
            --   %n = "stablehlo.reduce"(%input, %init) ({
            --     ^bb0(%argN: type, %argM: type):
            --       %p = stablehlo.add %argN, %argM : (type, type) -> type
            --       "stablehlo.return"(%p) : (type) -> ()
            --   }) {dimensions = array<i64: ...>} : (types) -> type
            prettyResultVids results <> " = \"stablehlo.reduce\"("
            <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
            <> " ("
            <> mconcat (map prettyRegion regions)
            <> ") "
            <> prettyAttrs (filter (not . isAppliesAttr) attrs)
            <> " : " <> prettyResultType operandTypes resultTypes
      where
        isAppliesAttr (AttrString "applies" _) = True
        isAppliesAttr _ = False
    pretty (Operation "stablehlo.convolution" operands operandTypes attrs regions results resultTypes) =
        -- Custom format for convolution:
        --   %r = stablehlo.convolution(%lhs, %rhs)
        --        dim_numbers = ..., window = {...}
        --        {batch_group_count = 1 : i64, ...}
        --        : (lhs_type, rhs_type) -> result_type
        prettyResultVids results <> " = stablehlo.convolution("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> prettyConvAttrs attrs
        <> " : " <> prettyResultType operandTypes resultTypes
        <> mconcat (map prettyRegion regions)
    pretty (Operation "stablehlo.dot_general" operands operandTypes attrs regions results resultTypes) =
        -- Custom format for dot_general:
        --   %r = stablehlo.dot_general %lhs, %rhs,
        --        batching_dims = [0] x [0],
        --        contracting_dims = [2] x [1]
        --        : (lhs_type, rhs_type) -> result_type
        prettyResultVids results <> " = stablehlo.dot_general "
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ","
        <> prettyDotGeneralAttrs attrs
        <> " : " <> prettyResultType operandTypes resultTypes
        <> mconcat (map prettyRegion regions)
    pretty (Operation "stablehlo.batch_norm_inference" operands operandTypes attrs regions results resultTypes) =
        -- Custom format: %r = stablehlo.batch_norm_inference %x, %scale, %offset, %mean, %variance
        --   <{epsilon = 1.0E-5 : f32, feature_index = 1 : i64}> : type
        prettyResultVids results <> " = stablehlo.batch_norm_inference "
        <> mconcat (intersperse (", ") (map valueRefBuilder operands))
        <> prettyBNAttrs attrs
        <> " : " <> prettyResultType operandTypes resultTypes
        <> mconcat (map prettyRegion regions)
    pretty (Operation "stablehlo.gather" operands operandTypes attrs regions results resultTypes) =
        -- Generic form (no custom assembly in this parser version).
        prettyResultVids results <> " = \"stablehlo.gather\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null regions then mempty else mconcat (map prettyRegion regions))
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.compare" operands operandTypes attrs regions results resultTypes) =
        -- Generic form for maximum parser compatibility.
        prettyResultVids results <> " = \"stablehlo.compare\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.slice" operands operandTypes attrs regions results resultTypes) =
        -- Generic form to maximise parser compatibility.
        prettyResultVids results <> " = \"stablehlo.slice\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null regions then mempty else mconcat (map prettyRegion regions))
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.pad" operands operandTypes attrs regions results resultTypes) =
        -- Generic form to maximise parser compatibility.
        prettyResultVids results <> " = \"stablehlo.pad\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null regions then mempty else mconcat (map prettyRegion regions))
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.dynamic_slice" operands operandTypes attrs regions results resultTypes) =
        -- Generic form to maximise parser compatibility.
        prettyResultVids results <> " = \"stablehlo.dynamic_slice\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null regions then mempty else mconcat (map prettyRegion regions))
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.transpose" operands operandTypes attrs regions results resultTypes) =
        -- Generic form with array<i64: ...> for permutation (PJRT v1.16.0 compat).
        let attrs' = map fixPermAttr attrs
        in prettyResultVids results <> " = \"stablehlo.transpose\"("
           <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
           <> (if null attrs' then mempty else " " <> prettyAttrs attrs')
           <> " : " <> prettyResultType operandTypes resultTypes
      where
        fixPermAttr (AttrIntList "permutation" vals) =
            AttrRaw $ "permutation = array<i64: " <> T.intercalate ", " (map (T.pack . show) vals) <> ">"
        fixPermAttr a = a
    pretty (Operation "stablehlo.reverse" operands operandTypes attrs regions results resultTypes) =
        -- Generic form with array<i64: ...> for dimensions (PJRT v1.16.0 compat).
        let attrs' = map fixRevAttr attrs
        in prettyResultVids results <> " = \"stablehlo.reverse\"("
           <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
           <> (if null attrs' then mempty else " " <> prettyAttrs attrs')
           <> " : " <> prettyResultType operandTypes resultTypes
      where
        fixRevAttr (AttrIntList "dimensions" vals) =
            AttrRaw $ "dimensions = array<i64: " <> T.intercalate ", " (map (T.pack . show) vals) <> ">"
        fixRevAttr a = a
    pretty (Operation "stablehlo.concatenate" operands operandTypes attrs regions results resultTypes) =
        -- Generic form (custom form syntax varies across parser versions).
        prettyResultVids results <> " = \"stablehlo.concatenate\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.iota" _ _ attrs _ results resultTypes) =
        -- Generic form (no operands).
        prettyResultVids results <> " = \"stablehlo.iota\"()"
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : () -> " <> prettyResults resultTypes
    pretty (Operation "stablehlo.sort" operands operandTypes attrs regions results resultTypes) =
        -- Generic form: regions must be wrapped in ( ) for PJRT parser compatibility.
        prettyResultVids results <> " = \"stablehlo.sort\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null regions then mempty else " (" <> mconcat (intersperse (", ") (map prettyRegion regions)) <> ")")
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.return" operands operandTypes _ regions _ _) =
        -- Generic form for the region terminator.
        "\"stablehlo.return\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands))
        <> ")"
        <> (if null regions then mempty else mconcat (map prettyRegion regions))
        <> " : "
        <> (if null operandTypes
            then "()"
            else "(" <> mconcat (intersperse (", ") (map pretty operandTypes)) <> ")")
        <> " -> ()"
    pretty (Operation "stablehlo.rng" operands operandTypes attrs regions results resultTypes) =
        -- Generic form for parser compatibility (no custom hook in some PJRT builds).
        prettyResultVids results <> " = \"stablehlo.rng\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation "stablehlo.rng_bit_generator" operands operandTypes attrs regions results resultTypes) =
        -- Generic form for parser compatibility.
        prettyResultVids results <> " = \"stablehlo.rng_bit_generator\"("
        <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultTypes
    pretty (Operation name operands operandTypes attrs regions results resultTypes) =
        if null regions
        then
            -- Existing custom-ish form for ops without regions.
            prettyResultVids results <> " = " <> fromText name
            <> (if null operands then mempty else " " <> mconcat (intersperse (fromText ", ") (map valueRefBuilder operands)))
            <> prettyAttrsForOp name attrs
            <> " : " <> prettyResultType operandTypes resultTypes
        else
            -- Generic assembly form for ops with regions (maximises parser compatibility).
            prettyResultVids results <> " = \"" <> fromText name <> "\""
            <> "(" <> mconcat (intersperse (", ") (map valueRefBuilder operands)) <> ")"
            <> " (" <> mconcat (intersperse (", ") (map prettyRegion regions)) <> ")"
            <> (if null attrs then mempty else " " <> prettyAttrs attrs)
            <> " : " <> prettyResultType operandTypes resultTypes

-- | Pretty-print operation attributes.
-- For 'stablehlo.constant' with a single 'AttrDenseElements' we print the
-- dense value directly (no braces).
-- For 'stablehlo.broadcast_in_dim' with a single 'AttrIntList' we use the
-- custom trailing format @, dims = [...]@.
-- For all other cases we use the standard MLIR attribute-dictionary syntax.
prettyAttrsForOp :: Text -> [Attribute] -> Builder
prettyAttrsForOp _ [] = mempty
prettyAttrsForOp _ [AttrDenseElements shp dt vals] =
    " dense<" <> denseElements shp dt vals <> ">"
prettyAttrsForOp "stablehlo.broadcast_in_dim" [AttrIntList _name vals] =
    ", dims = [" <> fromText (T.intercalate ", " (map (T.pack . show) vals)) <> "]"
prettyAttrsForOp _ attrs = " " <> prettyAttrs attrs

-- | Pretty-print attributes for 'stablehlo.reduce'.
-- If an 'applies' attribute is present we use the shorthand form;
-- otherwise we only emit the dimensions (the region carries the op).
prettyReduceAttrs :: [Attribute] -> Builder
prettyReduceAttrs attrs =
    let dims = lookupAttrIntList "dimensions" attrs
        op   = lookupAttrString "applies" attrs
        dimsText = " across dimensions = [" <> fromText (T.intercalate ", " (map (T.pack . show) dims)) <> "]"
    in if T.null op
       then dimsText
       else " applies " <> fromText op <> dimsText

lookupAttrIntList :: Text -> [Attribute] -> [Int64]
lookupAttrIntList name = foldr f []
  where
    f (AttrIntList n xs) acc | n == name = xs ++ acc
    f _ acc = acc

lookupAttrString :: Text -> [Attribute] -> Text
lookupAttrString name = foldr f ""
  where
    f (AttrString n s) acc | n == name = s <> acc
    f _ acc = acc

-- | Pretty-print attributes for 'stablehlo.convolution'.
-- Extracts 'dim_numbers' and 'window' from the custom string attributes,
-- then renders the remaining attrs in the standard dictionary.
prettyConvAttrs :: [Attribute] -> Builder
prettyConvAttrs attrs =
    let dimNums   = lookupAttrString "dim_numbers" attrs
        window    = lookupAttrString "window" attrs
        rest      = filter (not . isCustomConvAttr) attrs
        custom    = (if T.null dimNums then mempty else " dim_numbers = " <> fromText dimNums <> ",")
                 <> (if T.null window  then mempty else " window = " <> fromText window)
        dict      = if null rest then mempty else " " <> prettyAttrs rest
    in custom <> dict
  where
    isCustomConvAttr (AttrString "dim_numbers" _) = True
    isCustomConvAttr (AttrString "window" _)      = True
    isCustomConvAttr _                            = False

-- | Pretty-print attributes for 'stablehlo.dot_general'.
-- Extracts 'batching_dims' and 'contracting_dims' from the custom string attributes.
prettyDotGeneralAttrs :: [Attribute] -> Builder
prettyDotGeneralAttrs attrs =
    let batch       = lookupAttrString "batching_dims" attrs
        contract    = lookupAttrString "contracting_dims" attrs
        rest        = filter (not . isCustomDotAttr) attrs
        custom      = (if T.null batch    then mempty else "\n    batching_dims = " <> fromText batch <> ",")
                   <> (if T.null contract then mempty else "\n    contracting_dims = " <> fromText contract)
        dict        = if null rest then mempty else "\n    " <> prettyAttrs rest
    in custom <> dict
  where
    isCustomDotAttr (AttrString "batching_dims" _)    = True
    isCustomDotAttr (AttrString "contracting_dims" _) = True
    isCustomDotAttr _                                 = False

-- | Pretty-print attributes for 'stablehlo.batch_norm_inference'.
-- Uses the generic op format with <{...}> around the attributes.
prettyBNAttrs :: [Attribute] -> Builder
prettyBNAttrs attrs = " <{" <> mconcat (intersperse (", ") (map prettyAttr attrs)) <> ">}"

-- | Build the nested list syntax for a 'dense<...>' attribute.
-- Scalar: @0.0@, 1-D: @[1, 2]@, 2-D: @[[1, 2], [3, 4]]@, etc.
denseElements :: [Integer] -> DType -> [Double] -> Builder
denseElements []     dt [v] = fromText (T.pack (formatDenseVal dt v))
denseElements []     _  _   = error "denseElements: scalar mismatch"
denseElements [n]    dt xs  =
    "[" <> fromText (T.intercalate ", " (map (T.pack . formatDenseVal dt) (take (fromIntegral n) xs))) <> "]"
denseElements (n:ns) dt xs  =
    let chunkSize = product ns
        chunks    = chunksOf (fromIntegral chunkSize) (take (fromIntegral (n * chunkSize)) xs)
    in "[" <> mconcat (intersperse (", ") (map (denseElements ns dt) chunks)) <> "]"

formatDenseVal :: DType -> Double -> String
formatDenseVal I64 v = show (round v :: Integer)
formatDenseVal I32 v = show (round v :: Integer)
formatDenseVal I16 v = show (round v :: Integer)
formatDenseVal I8  v = show (round v :: Integer)
formatDenseVal UI64 v = show (round v :: Integer)
formatDenseVal UI32 v = show (round v :: Integer)
formatDenseVal UI16 v = show (round v :: Integer)
formatDenseVal UI8  v = show (round v :: Integer)
formatDenseVal Bool v = if v /= 0.0 then "true" else "false"
formatDenseVal _    v = show v

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf k xs = let (h, t) = splitAt k xs in h : chunksOf k t

-- | Pretty-print one or more result value IDs.
-- In standard MLIR, results are comma-separated without parens:
--   %0 = op(...)          -- one result
--   %0, %1 = op(...)      -- two results
prettyResultVids :: [ValueId] -> Builder
prettyResultVids []     = ""
prettyResultVids vs     =
    mconcat (intersperse (fromText ", ") (map valueRefBuilder vs))

-- | When an operation has no operands we print just the result type.
-- When it has operands we print (operandTypes) -> resultType.
-- For multi-result ops: (operandTypes) -> (resultType1, resultType2, ...)
prettyResultType :: [TensorType] -> [TensorType] -> Builder
prettyResultType [] [rt] = pretty rt
prettyResultType [] rts  = prettyResults rts
prettyResultType ots [rt] =
    "(" <> mconcat (intersperse (fromText ", ") (map pretty ots)) <> ") -> " <> pretty rt
prettyResultType ots rts =
    "(" <> mconcat (intersperse (fromText ", ") (map pretty ots)) <> ") -> " <> prettyResults rts

instance Pretty TensorType where
    pretty (TensorType [] dtype) =
        "tensor<" <> fromText (dtypeToText dtype) <> ">"
    pretty (TensorType shape dtype) =
        "tensor<" <> fromText dims <> "x" <> fromText (dtypeToText dtype) <> ">"
      where
        dims = T.intercalate "x" (map (T.pack . show) shape)

valueRefBuilder :: ValueId -> Builder
valueRefBuilder v = fromText (valueRef v)

prettyAttrs :: [Attribute] -> Builder
prettyAttrs attrs = "{" <> mconcat (intersperse (", ") (map prettyAttr attrs)) <> "}"

prettyAttr :: Attribute -> Builder
prettyAttr (AttrInt name val) =
    fromText name <> " = " <> fromText (T.pack (show val)) <> " : i64"
prettyAttr (AttrFloat name val) =
    fromText name <> " = " <> fromText (T.pack (show val)) <> " : f32"
prettyAttr (AttrBool name True) =
    fromText name <> " = true"
prettyAttr (AttrBool name False) =
    fromText name <> " = false"
prettyAttr (AttrString name s) =
    fromText name <> " = \"" <> fromText s <> "\""
prettyAttr (AttrIntList name vals) =
    fromText name <> " = [" <> fromText (T.intercalate ", " (map (T.pack . show) vals)) <> "]"
prettyAttr (AttrDenseElements shape dtype vals) =
    "value = dense<" <> denseElements shape dtype vals <> "> : " <> pretty (TensorType shape dtype)
prettyAttr (AttrDict pairs) =
    mconcat (intersperse (", ") (map prettyDictPair pairs))
  where
    prettyDictPair (k, v) = fromText k <> " = " <> prettyAttr v
prettyAttr (AttrRaw t) =
    fromText t

prettyRegion :: Region -> Builder
prettyRegion (Region blocks) =
    "{\n"
    <> mconcat (zipWith prettyBlock [0..] blocks)
    <> "  }"

prettyBlock :: Int -> Block -> Builder
prettyBlock idx (Block args ops) =
    "  ^bb" <> fromText (T.pack (show idx))
    <> (if null args then mempty else "(" <> mconcat (intersperse (", ") (map pretty args)) <> ")")
    <> ":\n"
    <> mconcat (map (\op -> "    " <> pretty op <> "\n") ops)

intersperse :: Builder -> [Builder] -> [Builder]
intersperse _   []     = []
intersperse _   [x]    = [x]
intersperse sep (x:xs) = x : map (sep <>) xs
