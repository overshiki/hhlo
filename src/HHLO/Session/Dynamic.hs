{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Dynamic-shape execution with automatic shape specialization.
--
-- This module provides 'compileDynamic' and 'runDynamicCompiled' which
-- work on both CPU and GPU backends by specialising the module to
-- concrete shapes at runtime.
--
-- Usage:
--
-- > let dm = dynamicModule "main"
-- >            [ TensorType [Nothing] F32 ]
-- >            $ \argTypes -> do
-- >                a <- anyArg (head argTypes)
-- >                b <- anyAdd a a
-- >                return (anyVid b, anyType b)
-- > compiled <- withCPU $ \sess -> compileDynamic sess dm
-- > [out] <- runDynamicCompiled compiled [input]
--
module HHLO.Session.Dynamic
    ( DynamicModule(..)
    , dynamicModule
    , DynamicCompiled
    , compileDynamic
    , runDynamicCompiled
    ) where

import Control.Monad.State
import Data.IORef
import Data.Int (Int64)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Vector.Storable as V

import HHLO.Core.Types
import HHLO.IR.AST
import HHLO.IR.Builder
import HHLO.Session
import HHLO.IR.Pretty (render)

-- | A dynamic module template that can be specialised to concrete shapes.
data DynamicModule = DynamicModule
    { dmName       :: !Text
    , dmArgTypes   :: ![TensorType]
    , dmBuildAction :: [TensorType] -> Builder (ValueId, TensorType)
    }

-- | Create a dynamic module template.
--
-- The builder function receives concrete argument types (with 'Nothing'
-- replaced by actual sizes) and must return the result value id and type.
dynamicModule :: Text -> [TensorType] -> ([TensorType] -> Builder (ValueId, TensorType)) -> DynamicModule
dynamicModule = DynamicModule

-- | A compiled dynamic module with a shape-specialisation cache.
data DynamicCompiled = DynamicCompiled
    { dcSession :: !Session
    , dcModule  :: !DynamicModule
    , dcCache   :: !(IORef (Map [TensorType] Compiled))
    }

-- | Substitute concrete sizes into a 'TensorType'.
-- 'Nothing' dimensions are replaced by the corresponding runtime size.
setShape :: TensorType -> [Int64] -> TensorType
setShape ttype shapes =
    ttype { ttShape = zipWith mergeDim (ttShape ttype) shapes }
  where
    mergeDim Nothing  s = Just (fromIntegral s)
    mergeDim (Just n) _ = Just n

-- | Compile a dynamic module.
--
-- Does not actually compile anything yet — compilation is deferred until
-- the first 'runDynamicCompiled' call with a concrete shape.
compileDynamic :: Session -> DynamicModule -> IO DynamicCompiled
compileDynamic sess dm = do
    ref <- newIORef Map.empty
    return $ DynamicCompiled sess dm ref

-- | Run a dynamic compiled module with concrete inputs.
-- If this shape combination has not been seen before, a static module
-- is built and compiled automatically.
runDynamicCompiled :: forall d. (KnownDType d, V.Storable (HostType d))
                   => DynamicCompiled -> [DynamicHostTensor d] -> IO [DynamicHostTensor d]
runDynamicCompiled dc inputs = do
    let concreteArgTypes = zipWith setShape (dmArgTypes $ dcModule dc) (map dhtShape inputs)
    cache <- readIORef (dcCache dc)
    compiled <- case Map.lookup concreteArgTypes cache of
        Just c  -> return c
        Nothing -> do
            let dm = dcModule dc
                ((resVid, resType), ops) = runBuilderRaw (dmBuildAction dm concreteArgTypes)
                args = zipWith (\i t -> FuncArg (T.pack ("arg" ++ show i)) t) [0::Int ..] concreteArgTypes
                func = Function (dmName dm) args [resType] [resVid] ops
                modu = Module [func]
            c <- compile (dcSession dc) modu
            atomicModifyIORef' (dcCache dc) $ \m -> (Map.insert concreteArgTypes c m, ())
            return c
    runDynamic (dcSession dc) compiled inputs
