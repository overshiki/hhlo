-- | Autograd-HHLO: Reverse-mode automatic differentiation for StableHLO.
--
-- This library provides 'grad' and 'vjp' combinators that transform HHLO
-- computation graphs into their gradients, producing new StableHLO modules
-- that compile via PJRT to CPU or GPU.
--
-- The primary safe entry points are 'gradModule' and 'vjpModule', which
-- produce standalone 'Module' values. The in-place 'grad' and 'vjp'
-- combinators can be used inside 'buildModule' for composability.
--
-- Example:
--
-- > import HHLO.ModuleBuilder
-- > import HHLO.Autograd
-- >
-- > -- f(x) = sum(x^2)
-- > -- grad f(x) = 2 * x
-- > gradMod = gradModule @'[3] @'F32 $ \x -> do
-- >     sq <- multiply x x
-- >     sumAll sq
module HHLO.Autograd
    ( module HHLO.Autograd.Core
    , module HHLO.Autograd.Grad
    , module HHLO.Autograd.Rules
    ) where

import HHLO.Autograd.Core
import HHLO.Autograd.Grad
import HHLO.Autograd.Rules
