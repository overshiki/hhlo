{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.Autograd.Rules
    ( backwardStep
    ) where

import Data.Int (Int64)
import Data.List (sortOn)
import Data.Maybe (isNothing)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

import HHLO.IR.AST
import HHLO.IR.Builder

import HHLO.Autograd.Core

-- ---------------------------------------------------------------------------
-- Main backward step dispatcher
-- ---------------------------------------------------------------------------

-- | Process one forward operation in reverse order, propagating cotangents.
backwardStep :: Map ValueId BTensor -> Operation -> Builder (Map ValueId BTensor)
backwardStep cmap op
    | all isNothing resultBars = return cmap
    | otherwise = case opName op of
        "stablehlo.add"          -> vjpAdd op resultBars cmap
        "stablehlo.subtract"     -> vjpSubtract op resultBars cmap
        "stablehlo.multiply"     -> vjpMultiply op resultBars cmap
        "stablehlo.divide"       -> vjpDivide op resultBars cmap
        "stablehlo.negate"       -> vjpNegate op resultBars cmap
        "stablehlo.exponential"  -> vjpExponential op resultBars cmap
        "stablehlo.log"          -> vjpLog op resultBars cmap
        "stablehlo.sqrt"         -> vjpSqrt op resultBars cmap
        "stablehlo.sine"         -> vjpSine op resultBars cmap
        "stablehlo.cosine"       -> vjpCosine op resultBars cmap
        "stablehlo.power"        -> vjpPower op resultBars cmap
        "stablehlo.reshape"      -> vjpReshape op resultBars cmap
        "stablehlo.transpose"    -> vjpTranspose op resultBars cmap
        "stablehlo.broadcast_in_dim" -> vjpBroadcastInDim op resultBars cmap
        "stablehlo.reduce"       -> vjpReduce op resultBars cmap
        "stablehlo.dot"          -> vjpDot op resultBars cmap
        "stablehlo.select"       -> vjpSelect op resultBars cmap
        "stablehlo.slice"        -> vjpSlice op resultBars cmap
        "stablehlo.pad"          -> vjpPad op resultBars cmap
        "stablehlo.concatenate"  -> vjpConcatenate op resultBars cmap
        "stablehlo.gather"       -> vjpGather op resultBars cmap
        "stablehlo.scatter"      -> vjpScatter op resultBars cmap
        "stablehlo.convert"      -> vjpConvert op resultBars cmap
        "stablehlo.abs"          -> vjpAbs op resultBars cmap
        "stablehlo.maximum"      -> vjpMaximum op resultBars cmap
        "stablehlo.minimum"      -> vjpMinimum op resultBars cmap
        "stablehlo.tanh"         -> vjpTanh op resultBars cmap
        "stablehlo.constant"     -> return cmap  -- constants have zero gradient
        "stablehlo.return"       -> return cmap  -- internal terminator
        "stablehlo.compare"      -> return cmap  -- comparisons have zero gradient
        "stablehlo.floor"        -> return cmap  -- non-differentiable
        "stablehlo.ceil"         -> return cmap  -- non-differentiable
        "stablehlo.sort"         -> error "autograd-hhlo: sort/topK is not differentiable"
        _ -> error $ T.unpack $ "autograd-hhlo: no VJP rule for " <> opName op
  where
    resultBars = map (\r -> Map.lookup r cmap) (opResults op)

-- ---------------------------------------------------------------------------
-- VJP helpers
-- ---------------------------------------------------------------------------

getResultBar :: [Maybe BTensor] -> Maybe BTensor
getResultBar [mb] = mb
getResultBar _    = Nothing

operandBT :: Operation -> Int -> BTensor
operandBT op i = BTensor (opOperands op !! i) (opOperandTypes op !! i)

-- ---------------------------------------------------------------------------
-- Arithmetic VJP rules
-- ---------------------------------------------------------------------------

vjpAdd :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpAdd op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            yVid = opOperands op !! 1
        cmap' <- accumulate cmap xVid bar
        accumulate cmap' yVid bar
    Nothing -> return cmap

vjpSubtract :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpSubtract op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            yVid = opOperands op !! 1
        negBar <- bnegate bar
        cmap' <- accumulate cmap xVid bar
        accumulate cmap' yVid negBar
    Nothing -> return cmap

vjpMultiply :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpMultiply op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
            y = operandBT op 1
        dx <- bmultiply bar y
        dy <- bmultiply bar x
        cmap' <- accumulate cmap (btVid x) dx
        accumulate cmap' (btVid y) dy
    Nothing -> return cmap

vjpDivide :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpDivide op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
            y = operandBT op 1
        -- dx = bar / y
        dx <- bdivide bar y
        -- dy = -bar * x / (y * y)
        negBar <- bnegate bar
        t1 <- bmultiply negBar x
        ySq <- bmultiply y y
        dy <- bdivide t1 ySq
        cmap' <- accumulate cmap (btVid x) dx
        accumulate cmap' (btVid y) dy
    Nothing -> return cmap

vjpNegate :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpNegate op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        negBar <- bnegate bar
        accumulate cmap (opOperands op !! 0) negBar
    Nothing -> return cmap

-- ---------------------------------------------------------------------------
-- Math VJP rules
-- ---------------------------------------------------------------------------

vjpExponential :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpExponential op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        -- Forward result y = exp(x), so dy/dx = exp(x) = y.
        -- We need the forward result value. For now, recompute exp(x).
        let x = operandBT op 0
        y <- bexp x
        dx <- bmultiply bar y
        accumulate cmap (btVid x) dx
    Nothing -> return cmap

vjpLog :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpLog op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
        -- dx = bar / x
        dx <- bdivide bar x
        accumulate cmap (btVid x) dx
    Nothing -> return cmap

vjpSqrt :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpSqrt op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
        -- dx = bar / (2 * sqrt(x))
        two <- bconstant (btType x) 2.0
        s <- bsqrt x
        t1 <- bmultiply two s
        dx <- bdivide bar t1
        accumulate cmap (btVid x) dx
    Nothing -> return cmap

vjpSine :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpSine op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
        c <- bcos x
        dx <- bmultiply bar c
        accumulate cmap (btVid x) dx
    Nothing -> return cmap

vjpCosine :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpCosine op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
        s <- bsin x
        negS <- bnegate s
        dx <- bmultiply bar negS
        accumulate cmap (btVid x) dx
    Nothing -> return cmap

vjpPower :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpPower op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
            y = operandBT op 1
        -- For z = x^y:
        -- dz/dx = y * x^(y-1)
        -- dz/dy = x^y * log(x)
        one <- bconstant (btType y) 1.0
        yMinus1 <- bsubtract y one
        xPow <- bpower x yMinus1
        dx <- bmultiply y xPow
        z <- bpower x y
        logX <- blog x
        dy <- bmultiply z logX
        dx' <- bmultiply bar dx
        dy' <- bmultiply bar dy
        cmap' <- accumulate cmap (btVid x) dx'
        accumulate cmap' (btVid y) dy'
    Nothing -> return cmap

-- Helper for power (not exported, used only inside VJP)
bpower :: BTensor -> BTensor -> Builder BTensor
bpower (BTensor x t) (BTensor y _) = do
    vid <- emitOp "stablehlo.power" [x, y] [t, t] [] t
    return (BTensor vid t)

-- ---------------------------------------------------------------------------
-- Shape manipulation VJP rules
-- ---------------------------------------------------------------------------

vjpReshape :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpReshape op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            xType = opOperandTypes op !! 0
        -- Gradient is a reshape back to the input shape.
        dx <- breshape bar xType
        accumulate cmap xVid dx
    Nothing -> return cmap

vjpTranspose :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpTranspose op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            xType = opOperandTypes op !! 0
        -- Extract permutation from attributes.
        perm <- case findPermutation (opAttributes op) of
            Just p  -> return p
            Nothing -> error "autograd-hhlo: transpose missing permutation attribute"
        -- Inverse permutation.
        let invPerm = invertPerm perm
        dx <- btranspose bar invPerm xType
        accumulate cmap xVid dx
    Nothing -> return cmap
  where
    findPermutation :: [Attribute] -> Maybe [Int64]
    findPermutation [] = Nothing
    findPermutation (AttrIntList "permutation" p : _) = Just p
    findPermutation (_ : attrs) = findPermutation attrs

    invertPerm :: [Int64] -> [Int64]
    invertPerm perm = map fst $ sortOn snd $ zip ([0 ..] :: [Int64]) perm

vjpBroadcastInDim :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpBroadcastInDim op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            xType = opOperandTypes op !! 0
            outType = head (opResultTypes op)
        -- Extract broadcast dims.
        dims <- case findDims (opAttributes op) of
            Just d  -> return d
            Nothing -> error "autograd-hhlo: broadcast_in_dim missing dims attribute"
        -- Gradient: reduce over the broadcasted dimensions.
        let outRank = length (ttShape outType) :: Int
            broadcastDims = map fromIntegral dims :: [Int]
            reduceDims = filter (`notElem` broadcastDims) [0 .. outRank - 1]
        if null reduceDims
            then accumulate cmap xVid bar
            else do
                dx <- breduceSum bar reduceDims xType
                accumulate cmap xVid dx
    Nothing -> return cmap
  where
    findDims [] = Nothing
    findDims (AttrIntList "dims" d : _) = Just d
    findDims (_ : attrs) = findDims attrs

-- ---------------------------------------------------------------------------
-- Reduction VJP rules
-- ---------------------------------------------------------------------------

vjpReduce :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpReduce op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            xType = opOperandTypes op !! 0
        -- We only support sum reductions for now.
        -- Detect sum by looking for stablehlo.add in the reduction region.
        let isSum = any regionHasAdd (opRegions op)
            regionHasAdd (Region blocks) = any blockHasAdd blocks
            blockHasAdd (Block _ ops) = any (\o -> opName o == "stablehlo.add") ops
        if not isSum
            then error "autograd-hhlo: only sum reductions are supported"
            else do
                dims <- case findDimensions (opAttributes op) of
                    Just d  -> return $ map fromIntegral d
                    Nothing -> error "autograd-hhlo: reduce missing dimensions attribute"
                -- Gradient: broadcast the cotangent back to the input shape.
                dx <- broadcastLike bar xType dims
                accumulate cmap xVid dx
    Nothing -> return cmap
  where
    findDimensions [] = Nothing
    findDimensions (AttrRaw raw : _) =
        if "dimensions" `T.isPrefixOf` T.strip raw
        then Just $ parseIntList raw
        else Nothing
    findDimensions (AttrIntList "dimensions" dims : _) = Just dims
    findDimensions (_ : attrs) = findDimensions attrs

    parseIntList raw =
        let inner = T.takeWhile (/= '>') $ T.dropWhile (/= '<') raw
            withoutLt = T.dropWhile (== '<') inner
            withoutPrefix = if "i64:" `T.isPrefixOf` withoutLt then T.drop 4 withoutLt else withoutLt
            nums = T.splitOn "," withoutPrefix
        in map (read . T.unpack . T.strip) nums

    -- Broadcast a reduced cotangent back to the original shape.
    broadcastLike :: BTensor -> TensorType -> [Int] -> Builder BTensor
    broadcastLike bar inType dims = do
        let inRank = length (ttShape inType)
            -- Build broadcast dims: map each reduced dim to its position.
            -- For reduce over dims [0,1] of a 3-D tensor, we broadcast
            -- the scalar/shape to the original shape.
            -- We use broadcast_in_dim with dims = remaining_dims.
            remainingDims = filter (`notElem` dims) [0 .. inRank - 1]
        if length remainingDims == inRank
            then return bar  -- No reduction happened
            else do
                bbroadcastInDim bar (map fromIntegral remainingDims) inType

-- ---------------------------------------------------------------------------
-- Linear algebra VJP rules
-- ---------------------------------------------------------------------------

vjpDot :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpDot op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
            y = operandBT op 1
            xType = btType x
            yType = btType y
        -- For C = A @ B:
        -- dA = dC @ B^T
        -- dB = A^T @ dC
        -- We need to transpose the appropriate dimensions.
        -- For simplicity, assume standard 2-D matmul.
        let xShape = ttShape xType
            yShape = ttShape yType
        if length xShape == 2 && length yShape == 2
            then do
                bT <- btranspose y [1, 0] (TensorType (reverse $ ttShape yType) (ttDType yType))
                aT <- btranspose x [1, 0] (TensorType (reverse xShape) (ttDType xType))
                da <- bdot bar bT xType
                db <- bdot aT bar yType
                cmap' <- accumulate cmap (btVid x) da
                accumulate cmap' (btVid y) db
            else
                error "autograd-hhlo: dot VJP only supports 2-D matrices for now"
    Nothing -> return cmap

-- ---------------------------------------------------------------------------
-- Selection VJP rule
-- ---------------------------------------------------------------------------

vjpSelect :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpSelect op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let predVid  = opOperands op !! 0
            trueVid  = opOperands op !! 1
            falseVid = opOperands op !! 2
            predType = opOperandTypes op !! 0
            valType  = opOperandTypes op !! 1
        zero <- bconstant valType 0.0
        -- dx = select(pred, bar, 0)
        dx <- bselect (BTensor predVid predType) bar zero valType
        -- dy = select(pred, 0, bar)
        dy <- bselect (BTensor predVid predType) zero bar valType
        cmap' <- accumulate cmap trueVid dx
        accumulate cmap' falseVid dy
    Nothing -> return cmap

-- ---------------------------------------------------------------------------
-- Slice VJP rule
-- ---------------------------------------------------------------------------

vjpSlice :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpSlice op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            xType = opOperandTypes op !! 0
            (start, limit, stride) = parseSliceAttrs (opAttributes op)
            xShape = ttShape xType
            rank = length xShape
            low = start
            isUnitStride = all (== 1) stride
        if isUnitStride
            then do
                let high' = map fromIntegral (zipWith (-) xShape (map fromIntegral limit :: [Integer])) :: [Int64]
                    interior = replicate rank (0 :: Int64)
                zero <- bconstant xType 0.0
                dx <- bpad bar zero low high' interior xType
                accumulate cmap xVid dx
            else do
                -- General stride: use interior padding = stride - 1
                let start' = map fromIntegral start :: [Integer]
                    limit' = map fromIntegral limit :: [Integer]
                    stride' = map fromIntegral stride :: [Integer]
                    high' = map fromIntegral (zipWith (-) xShape (zipWith (+) start' (zipWith (*) (zipWith (-) limit' start') stride'))) :: [Int64]
                    interior = map (\s -> max 0 (s - 1)) stride :: [Int64]
                zero <- bconstant xType 0.0
                dx <- bpad bar zero low high' interior xType
                accumulate cmap xVid dx
    Nothing -> return cmap
  where
    parseSliceAttrs :: [Attribute] -> ([Int64], [Int64], [Int64])
    parseSliceAttrs attrs =
        let s = findAttr "start_indices" attrs
            l = findAttr "limit_indices" attrs
            st = findAttr "strides" attrs
        in (s, l, st)

    findAttr :: Text -> [Attribute] -> [Int64]
    findAttr _ [] = error "autograd-hhlo: slice missing attribute"
    findAttr name (AttrRaw raw : rest) =
        let key = name <> " = array<i64:"
        in if key `T.isInfixOf` raw
            then parseIntList raw
            else findAttr name rest
    findAttr name (_ : rest) = findAttr name rest

    parseIntList :: Text -> [Int64]
    parseIntList raw =
        let inner = T.takeWhile (/= '>') $ T.dropWhile (/= '<') raw
            withoutLt = T.dropWhile (== '<') inner
            withoutPrefix = if "i64:" `T.isPrefixOf` withoutLt then T.drop 4 withoutLt else withoutLt
            nums = T.splitOn "," withoutPrefix
        in map ((fromIntegral :: Integer -> Int64) . read . T.unpack . T.strip) nums

-- ---------------------------------------------------------------------------
-- Concatenate VJP rule
-- ---------------------------------------------------------------------------

vjpConcatenate :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpConcatenate op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let inputVids = opOperands op
            inputTypes = opOperandTypes op
        dim <- case findConcatDim (opAttributes op) of
            Just d  -> return d
            Nothing -> error "autograd-hhlo: concatenate missing dimension attribute"
        -- Split bar along concat dimension into slices matching each input
        let inputSizes = map (!! dim) (map ttShape inputTypes)
        splitAndAccumulate bar dim 0 inputVids inputTypes inputSizes cmap
    Nothing -> return cmap
  where
    findConcatDim :: [Attribute] -> Maybe Int
    findConcatDim [] = Nothing
    findConcatDim (AttrInt "dimension" d : _) = Just (fromIntegral d)
    findConcatDim (_ : rest) = findConcatDim rest

    splitAndAccumulate :: BTensor -> Int -> Integer -> [ValueId] -> [TensorType] -> [Integer] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
    splitAndAccumulate _ _ _ [] [] [] acc = return acc
    splitAndAccumulate bar dim offset (vid:vids) (itype:itypes) (sz:szs) acc = do
        let shape = ttShape itype
            start = replicate (length shape) (0 :: Integer)
            limit = zipWith (\i s -> if i == dim then offset + s else s) [0..] shape
            stride = replicate (length shape) (1 :: Integer)
        piece <- bslice bar (map fromIntegral start) (map fromIntegral limit) (map fromIntegral stride) itype
        acc' <- accumulate acc vid piece
        splitAndAccumulate bar dim (offset + sz) vids itypes szs acc'
    splitAndAccumulate _ _ _ _ _ _ _ = error "autograd-hhlo: concatenate operand mismatch"

-- ---------------------------------------------------------------------------
-- Pad VJP rule
-- ---------------------------------------------------------------------------

vjpPad :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpPad op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            xType = opOperandTypes op !! 0
        (low, _high, interior) <- parsePadAttrs (opAttributes op)
        let xShape = ttShape xType
            stride = map (+ 1) interior :: [Int64]
            limit = zipWith3 (\l s sz -> l + sz * s) low stride (map fromIntegral xShape) :: [Int64]
        dx <- bslice bar low limit stride xType
        accumulate cmap xVid dx
    Nothing -> return cmap
  where
    parsePadAttrs :: [Attribute] -> Builder ([Int64], [Int64], [Int64])
    parsePadAttrs attrs =
        case (findAttr "edge_padding_low" attrs,
              findAttr "edge_padding_high" attrs,
              findAttr "interior_padding" attrs) of
            (Just l, Just h, Just i) -> return (l, h, i)
            _ -> error "autograd-hhlo: pad missing padding attributes"

    findAttr :: Text -> [Attribute] -> Maybe [Int64]
    findAttr _ [] = Nothing
    findAttr name (AttrRaw raw : rest) =
        let key = name <> " = array<i64:"
        in if key `T.isInfixOf` raw
            then Just (parseIntList raw)
            else findAttr name rest
    findAttr name (_ : rest) = findAttr name rest

    parseIntList :: Text -> [Int64]
    parseIntList raw =
        let inner = T.takeWhile (/= '>') $ T.dropWhile (/= '<') raw
            withoutLt = T.dropWhile (== '<') inner
            withoutPrefix = if "i64:" `T.isPrefixOf` withoutLt then T.drop 4 withoutLt else withoutLt
            nums = T.splitOn "," withoutPrefix
        in map ((fromIntegral :: Integer -> Int64) . read . T.unpack . T.strip) nums

-- ---------------------------------------------------------------------------
-- Gather / Scatter VJP rules (stubs)
-- ---------------------------------------------------------------------------

vjpGather :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpGather _ _ _ = error "autograd-hhlo: gather VJP not yet implemented"

vjpScatter :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpScatter _ _ _ = error "autograd-hhlo: scatter VJP not yet implemented"

-- ---------------------------------------------------------------------------
-- Convert VJP rule
-- ---------------------------------------------------------------------------

vjpConvert :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpConvert op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let xVid = opOperands op !! 0
            xType = opOperandTypes op !! 0
        dx <- bconvert bar xType
        accumulate cmap xVid dx
    Nothing -> return cmap

-- ---------------------------------------------------------------------------
-- Abs VJP rule
-- ---------------------------------------------------------------------------

vjpAbs :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpAbs op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
        absX <- babs x
        dx <- bdivide x absX
        dx' <- bmultiply bar dx
        accumulate cmap (btVid x) dx'
    Nothing -> return cmap

-- ---------------------------------------------------------------------------
-- Maximum / Minimum VJP rules
-- ---------------------------------------------------------------------------

vjpMaximum :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpMaximum op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
            y = operandBT op 1
        predGE <- bcompareGE x y
        zero <- bconstant (btType x) 0.0
        dx <- bselect predGE bar zero (btType x)
        dy <- bselect predGE zero bar (btType x)
        cmap' <- accumulate cmap (btVid x) dx
        accumulate cmap' (btVid y) dy
    Nothing -> return cmap

vjpMinimum :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpMinimum op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
            y = operandBT op 1
        predLE <- bcompareGE y x
        zero <- bconstant (btType x) 0.0
        dx <- bselect predLE bar zero (btType x)
        dy <- bselect predLE zero bar (btType x)
        cmap' <- accumulate cmap (btVid x) dx
        accumulate cmap' (btVid y) dy
    Nothing -> return cmap

-- ---------------------------------------------------------------------------
-- Tanh VJP rule
-- ---------------------------------------------------------------------------

vjpTanh :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpTanh op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
        tanhX <- btanh x
        tanhSq <- bmultiply tanhX tanhX
        one <- bconstant (btType x) 1.0
        oneMinus <- bsubtract one tanhSq
        dx <- bmultiply bar oneMinus
        accumulate cmap (btVid x) dx
    Nothing -> return cmap
