{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where
import qualified Data.Vector.Storable as V
import HHLO.EDSL.Dynamic
import HHLO.IR.AST (TensorType(..), FuncArg(..))
import HHLO.IR.Builder (moduleFromBuilderDynamic)
import HHLO.Session
import HHLO.Core.Types (DType(..))

main :: IO ()
main = do
    let modu = moduleFromBuilderDynamic "main"
            [ FuncArg "a" (TensorType [Nothing] F32) ]
            (TensorType [Nothing] F32)
            $ do
                a <- anyArg (TensorType [Nothing] F32)
                b <- anyAdd a a
                return (anyVid b)
    withGPU $ \sess -> do
        compiled <- compile sess modu
        let input4 = dynamicHostFromVector @'F32 (V.fromList [10.0, 20.0, 30.0, 40.0]) [4]
        [out4] <- runDynamic sess compiled [input4]
        print (V.toList (dhtData out4))
