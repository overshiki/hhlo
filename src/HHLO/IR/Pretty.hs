{-# LANGUAGE OverloadedStrings #-}

module HHLO.IR.Pretty
    ( Pretty(..)
    , render
    , renderLazy
    ) where

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
    pretty (Function name args result ops) =
        fromText "func.func @" <> fromText name <> "("
        <> mconcat (intersperse (fromText ", ") (map pretty args))
        <> ") -> " <> pretty result <> " {\n"
        <> mconcat (map ((<> "\n") . ("    " <>)) (map pretty ops))
        <> returnLine ops result
        <> "}"

returnLine :: [Operation] -> TensorType -> Builder
returnLine []     result = "    return : " <> pretty result <> "\n"
returnLine ops    result =
    let lastOp = last ops
    in "    return " <> valueRefBuilder (opResult lastOp) <> " : " <> pretty result <> "\n"

instance Pretty FuncArg where
    pretty (FuncArg name t) =
        fromText "%" <> fromText name <> ": " <> pretty t

instance Pretty Operation where
    pretty (Operation name operands operandTypes attrs result resultType) =
        valueRefBuilder result <> " = " <> fromText name
        <> (if null operands then mempty else " " <> mconcat (intersperse (fromText ", ") (map valueRefBuilder operands)))
        <> (if null attrs then mempty else " " <> prettyAttrs attrs)
        <> " : " <> prettyResultType operandTypes resultType

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
    "value = dense<[" <> fromText (T.intercalate ", " (map (T.pack . show) vals)) <> "]> : " <> pretty (TensorType shape dtype)
prettyAttr (AttrDict pairs) =
    mconcat (intersperse (", ") (map prettyDictPair pairs))
  where
    prettyDictPair (k, v) = fromText k <> " = " <> prettyAttr v

intersperse :: Builder -> [Builder] -> [Builder]
intersperse _   []     = []
intersperse _   [x]    = [x]
intersperse sep (x:xs) = x : map (sep <>) xs
