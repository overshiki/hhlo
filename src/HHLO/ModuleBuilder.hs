{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Auto-argument module construction.
--
-- Build StableHLO modules without manual 'FuncArg' declarations or 'natVal'
-- boilerplate. The input/output arity is specified via 'TypeApplications';
-- GHC selects the matching 'ModuleBuilder' instance.
--
-- > kmeansModule = buildModule @2 @1 "kmeans" $ \dat initC -> do
-- >     lloyd dat initC
--
-- ⚠️ 'TypeApplications' are mandatory: you must write @buildModule @nIn @nOut@.
-- ⚠️ Lambda parameters must be used in operations so GHC can infer their
--    'Tensor' type. Unused parameters require explicit type annotations.
module HHLO.ModuleBuilder
    ( buildModule
    , ModuleBuilder
    ) where

import GHC.TypeLits
import Data.Proxy
import Data.Text (Text)

import HHLO.Core.Types
import HHLO.IR.AST (Module, FuncArg(..))
import HHLO.IR.Builder

-- | Typeclass dispatching module construction by input and output arity.
-- Each instance auto-generates 'FuncArg' declarations and wires up 'arg' calls.
class ModuleBuilder (nIn :: Nat) (nOut :: Nat) f where
    buildModule' :: Text -> f -> Module

-- ---------------------------------------------------------------------------
-- 1 input, 1 output
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape sr, KnownDType dr
         ) => ModuleBuilder 1 1 (Tensor s1 d1 -> Builder (Tensor sr dr)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            builder = do
                t0 <- arg @s1 @d1
                f t0
        in moduleFromBuilder @sr @dr name [arg0] builder

-- ---------------------------------------------------------------------------
-- 1 input, 2 outputs
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape sr1, KnownDType dr1
         , KnownShape sr2, KnownDType dr2
         ) => ModuleBuilder 1 2 (Tensor s1 d1 -> Builder (Tuple2 sr1 dr1 sr2 dr2)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            builder = do
                t0 <- arg @s1 @d1
                f t0
        in moduleFromBuilder2 name [arg0] builder

-- ---------------------------------------------------------------------------
-- 1 input, 3 outputs
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape sr1, KnownDType dr1
         , KnownShape sr2, KnownDType dr2
         , KnownShape sr3, KnownDType dr3
         ) => ModuleBuilder 1 3 (Tensor s1 d1 -> Builder (Tuple3 sr1 dr1 sr2 dr2 sr3 dr3)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            builder = do
                t0 <- arg @s1 @d1
                f t0
        in moduleFromBuilder3 name [arg0] builder

-- ---------------------------------------------------------------------------
-- 2 inputs, 1 output
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape s2, KnownDType d2
         , KnownShape sr, KnownDType dr
         ) => ModuleBuilder 2 1 (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tensor sr dr)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            arg1 = FuncArg "arg1" (tensorType (Proxy @s2) (Proxy @d2))
            builder = do
                t0 <- arg @s1 @d1
                t1 <- arg @s2 @d2
                f t0 t1
        in moduleFromBuilder @sr @dr name [arg0, arg1] builder

-- ---------------------------------------------------------------------------
-- 2 inputs, 2 outputs
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape s2, KnownDType d2
         , KnownShape sr1, KnownDType dr1
         , KnownShape sr2, KnownDType dr2
         ) => ModuleBuilder 2 2 (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tuple2 sr1 dr1 sr2 dr2)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            arg1 = FuncArg "arg1" (tensorType (Proxy @s2) (Proxy @d2))
            builder = do
                t0 <- arg @s1 @d1
                t1 <- arg @s2 @d2
                f t0 t1
        in moduleFromBuilder2 name [arg0, arg1] builder

-- ---------------------------------------------------------------------------
-- 2 inputs, 3 outputs
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape s2, KnownDType d2
         , KnownShape sr1, KnownDType dr1
         , KnownShape sr2, KnownDType dr2
         , KnownShape sr3, KnownDType dr3
         ) => ModuleBuilder 2 3 (Tensor s1 d1 -> Tensor s2 d2 -> Builder (Tuple3 sr1 dr1 sr2 dr2 sr3 dr3)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            arg1 = FuncArg "arg1" (tensorType (Proxy @s2) (Proxy @d2))
            builder = do
                t0 <- arg @s1 @d1
                t1 <- arg @s2 @d2
                f t0 t1
        in moduleFromBuilder3 name [arg0, arg1] builder

-- ---------------------------------------------------------------------------
-- 3 inputs, 1 output
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape s2, KnownDType d2
         , KnownShape s3, KnownDType d3
         , KnownShape sr, KnownDType dr
         ) => ModuleBuilder 3 1 (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tensor sr dr)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            arg1 = FuncArg "arg1" (tensorType (Proxy @s2) (Proxy @d2))
            arg2 = FuncArg "arg2" (tensorType (Proxy @s3) (Proxy @d3))
            builder = do
                t0 <- arg @s1 @d1
                t1 <- arg @s2 @d2
                t2 <- arg @s3 @d3
                f t0 t1 t2
        in moduleFromBuilder @sr @dr name [arg0, arg1, arg2] builder

-- ---------------------------------------------------------------------------
-- 3 inputs, 2 outputs
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape s2, KnownDType d2
         , KnownShape s3, KnownDType d3
         , KnownShape sr1, KnownDType dr1
         , KnownShape sr2, KnownDType dr2
         ) => ModuleBuilder 3 2 (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tuple2 sr1 dr1 sr2 dr2)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            arg1 = FuncArg "arg1" (tensorType (Proxy @s2) (Proxy @d2))
            arg2 = FuncArg "arg2" (tensorType (Proxy @s3) (Proxy @d3))
            builder = do
                t0 <- arg @s1 @d1
                t1 <- arg @s2 @d2
                t2 <- arg @s3 @d3
                f t0 t1 t2
        in moduleFromBuilder2 name [arg0, arg1, arg2] builder

-- ---------------------------------------------------------------------------
-- 3 inputs, 3 outputs
-- ---------------------------------------------------------------------------

instance ( KnownShape s1, KnownDType d1
         , KnownShape s2, KnownDType d2
         , KnownShape s3, KnownDType d3
         , KnownShape sr1, KnownDType dr1
         , KnownShape sr2, KnownDType dr2
         , KnownShape sr3, KnownDType dr3
         ) => ModuleBuilder 3 3 (Tensor s1 d1 -> Tensor s2 d2 -> Tensor s3 d3 -> Builder (Tuple3 sr1 dr1 sr2 dr2 sr3 dr3)) where
    buildModule' name f =
        let arg0 = FuncArg "arg0" (tensorType (Proxy @s1) (Proxy @d1))
            arg1 = FuncArg "arg1" (tensorType (Proxy @s2) (Proxy @d2))
            arg2 = FuncArg "arg2" (tensorType (Proxy @s3) (Proxy @d3))
            builder = do
                t0 <- arg @s1 @d1
                t1 <- arg @s2 @d2
                t2 <- arg @s3 @d3
                f t0 t1 t2
        in moduleFromBuilder3 name [arg0, arg1, arg2] builder

-- ---------------------------------------------------------------------------
-- Polymorphic entry point
-- ---------------------------------------------------------------------------

-- | Build a 'Module' from a typed function with auto-generated 'FuncArg's.
--
-- Usage: @buildModule @inputs @outputs "name" $ \arg1 arg2 -> ...@
--
-- The @inputs@ and @outputs@ parameters are type-level 'Nat's specifying
-- the arity of the function. GHC selects the matching 'ModuleBuilder'
-- instance based on these numbers and the lambda's type.
--
-- The name argument is used as the function name in the generated MLIR.
-- For PJRT compatibility the function should be named @"main"@.
buildModule :: forall nIn nOut f. ModuleBuilder nIn nOut f => Text -> f -> Module
buildModule _name = buildModule' @nIn @nOut "main"
