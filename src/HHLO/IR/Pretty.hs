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
import HHLO.Core.Types (dtypeToText)

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

returnLine :: [ValueId] -> [TensorType] -> Builder
returnLine []     results = "    return : " <> prettyResults results <> "\n"
returnLine vids   results =
    let refs = mconcat (intersperse (fromText ", ") (map valueRefBuilder vids))
    in "    return " <> refs <> " : " <> prettyResults results <> "\n"

instance Pretty FuncArg where
    pretty (FuncArg name t) =
        fromText "%" <> fromText name <> ": " <> pretty t

instance Pretty Operation where
    pretty (Operation "stablehlo.reduce" operands operandTypes attrs regions result resultType) =
        -- Special format for reduce with 'applies' shorthand:
        --   %n = stablehlo.reduce(%input init: %init) applies stablehlo.add
        --        across dimensions = [0] : (input_type, init_type) -> result_type
        valueRefBuilder result <> " = stablehlo.reduce("
        <> valueRefBuilder (operands !! 0) <> " init: " <> valueRefBuilder (operands !! 1) <> ")"
        <> prettyReduceAttrs attrs
        <> " : " <> prettyResultType operandTypes resultType
        <> mconcat (map prettyRegion regions)
    pretty (Operation name operands operandTypes attrs regions result resultType) =
        -- stablehlo.return has no result value
        (if name == "stablehlo.return" then mempty else valueRefBuilder result <> " = ")
        <> fromText name
        <> (if null operands then mempty else " " <> mconcat (intersperse (fromText ", ") (map valueRefBuilder operands)))
        <> prettyAttrsForOp name attrs
        <> " : " <> prettyResultType operandTypes resultType
        <> mconcat (map prettyRegion regions)

-- | Pretty-print operation attributes.
-- For 'stablehlo.constant' with a single 'AttrDenseElements' we print the
-- dense value directly (no braces).
-- For 'stablehlo.broadcast_in_dim' with a single 'AttrIntList' we use the
-- custom trailing format @, dims = [...]@.
-- For all other cases we use the standard MLIR attribute-dictionary syntax.
prettyAttrsForOp :: Text -> [Attribute] -> Builder
prettyAttrsForOp _ [] = mempty
prettyAttrsForOp _ [AttrDenseElements shp _dt vals] =
    " dense<" <> denseElements shp vals <> ">"
prettyAttrsForOp "stablehlo.broadcast_in_dim" [AttrIntList _name vals] =
    ", dims = [" <> fromText (T.intercalate ", " (map (T.pack . show) vals)) <> "]"
prettyAttrsForOp _ attrs = " " <> prettyAttrs attrs

-- | Pretty-print attributes for 'stablehlo.reduce' using the 'applies' shorthand.
prettyReduceAttrs :: [Attribute] -> Builder
prettyReduceAttrs attrs =
    let dims = lookupAttrIntList "dimensions" attrs
        op   = lookupAttrString "applies" attrs
    in " applies " <> fromText op <> " across dimensions = [" <> fromText (T.intercalate ", " (map (T.pack . show) dims)) <> "]"

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

-- | Build the nested list syntax for a 'dense<...>' attribute.
-- Scalar: @0.0@, 1-D: @[1, 2]@, 2-D: @[[1, 2], [3, 4]]@, etc.
denseElements :: [Integer] -> [Double] -> Builder
denseElements []     [v] = fromText (T.pack (show v))
denseElements []     _   = error "denseElements: scalar mismatch"
denseElements [n]    xs  =
    "[" <> fromText (T.intercalate ", " (map (T.pack . show) (take (fromIntegral n) xs))) <> "]"
denseElements (n:ns) xs  =
    let chunkSize = product ns
        chunks    = chunksOf (fromIntegral chunkSize) (take (fromIntegral (n * chunkSize)) xs)
    in "[" <> mconcat (intersperse (", ") (map (denseElements ns) chunks)) <> "]"

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf k xs = let (h, t) = splitAt k xs in h : chunksOf k t

-- | When an operation has no operands we print just the result type.
-- When it has operands we print (operandTypes) -> resultType.
prettyResultType :: [TensorType] -> TensorType -> Builder
prettyResultType [] rt = pretty rt
prettyResultType ots rt =
    "(" <> mconcat (intersperse (fromText ", ") (map pretty ots)) <> ") -> " <> pretty rt

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
    "value = dense<" <> denseElements shape vals <> "> : " <> pretty (TensorType shape dtype)
prettyAttr (AttrDict pairs) =
    mconcat (intersperse (", ") (map prettyDictPair pairs))
  where
    prettyDictPair (k, v) = fromText k <> " = " <> prettyAttr v

prettyRegion :: Region -> Builder
prettyRegion (Region blocks) =
    " (" <> mconcat (map prettyBlock blocks) <> ")"

prettyBlock :: Block -> Builder
prettyBlock (Block args ops) =
    "{\n"
    <> "  ^bb0"
    <> (if null args then mempty else "(" <> mconcat (intersperse (", ") (map pretty args)) <> ")")
    <> ":\n"
    <> mconcat (map ((<> "\n") . ("    " <>)) (map pretty ops))
    <> "  }"

intersperse :: Builder -> [Builder] -> [Builder]
intersperse _   []     = []
intersperse _   [x]    = [x]
intersperse sep (x:xs) = x : map (sep <>) xs
