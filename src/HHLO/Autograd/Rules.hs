{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module HHLO.Autograd.Rules
    ( backwardStep
    ) where

import Data.Int (Int64)
import Data.List (sortOn, zipWith4)
import Data.Maybe (fromMaybe, isNothing, listToMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Debug.Trace (trace)

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
        "stablehlo.reduce_window" -> vjpReduceWindow op resultBars cmap
        "stablehlo.convolution"   -> vjpConvolution op resultBars cmap
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
    findDims (AttrIntList "broadcast_dimensions" d : _) = Just d
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
                let zeroType = TensorType [] (ttDType xType)
                zero <- bconstant zeroType 0.0
                dx <- bpad bar zero low high' interior xType
                accumulate cmap xVid dx
            else do
                -- General stride: use interior padding = stride - 1
                let start' = map fromIntegral start :: [Integer]
                    limit' = map fromIntegral limit :: [Integer]
                    stride' = map fromIntegral stride :: [Integer]
                    -- Number of elements in the sliced bar per dimension:
                    -- n = ceil((limit - start) / stride)
                    n = zipWith3 (\l s st -> (l - s + st - 1) `div` st) limit' start' stride' :: [Integer]
                    -- Padded size = start + n * stride - stride + 1 + high = xShape
                    -- => high = xShape - start - n * stride + stride - 1
                    high' = zipWith3 (\x (s, st) n_ -> fromIntegral (x - s - n_ * st + st - 1)) xShape (zip start' stride') n :: [Int64]
                    interior = map (\s -> max 0 (s - 1)) stride :: [Int64]
                    zeroType = TensorType [] (ttDType xType)
                zero <- bconstant zeroType 0.0
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
    findAttr name (AttrIntList n v : rest)
        | n == name = v
        | otherwise = findAttr name rest
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
            start = zipWith (\i _ -> if i == dim then offset else 0) [0..] shape
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
            limit = zipWith3 (\l s sz -> l + (sz - 1) * s + 1) low stride (map fromIntegral xShape) :: [Int64]
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
    findAttr name (AttrIntList n v : rest)
        | n == name = Just v
        | otherwise = findAttr name rest
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

-- ---------------------------------------------------------------------------
-- reduce_window VJP rule
-- ---------------------------------------------------------------------------

vjpReduceWindow :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpReduceWindow op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let x = operandBT op 0
            xType = btType x
            xVid  = btVid x
            xShape = ttShape xType
            rank = length xShape

        -- Detect reduction type by inspecting the region.
        let isAdd = any regionHasAdd (opRegions op)
            isMax = any regionHasMax (opRegions op)
            regionHasAdd (Region blocks) = any blockHasAdd blocks
            blockHasAdd (Block _ ops) = any (\o -> opName o == "stablehlo.add") ops
            regionHasMax (Region blocks) = any blockHasMax blocks
            blockHasMax (Block _ ops) = any (\o -> opName o == "stablehlo.maximum") ops

        -- Parse window attributes.
        let windowDims = findRawIntList "window_dimensions" (opAttributes op)
            strides    = findRawIntList "window_strides" (opAttributes op)
            padding    = findPadding (opAttributes op)

        if isAdd
            then vjpReduceWindowAdd bar xVid xType windowDims strides padding rank xShape cmap
            else if isMax
                then vjpReduceWindowMax bar x xType windowDims strides padding rank xShape cmap
                else error "autograd-hhlo: reduce_window VJP only supports add and maximum reductions"
    Nothing -> return cmap

vjpReduceWindowAdd :: BTensor -> ValueId -> TensorType -> [Int64] -> [Int64] -> [[Int64]] -> Int -> [Integer] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpReduceWindowAdd bar xVid xType windowDims strides padding _rank xShape cmap = do
    -- Only non-overlapping windows with VALID padding.
    let isNonOverlapping = and (zipWith (==) windowDims strides)
        isValid = all (all (== 0)) padding
    if not (isNonOverlapping && isValid)
        then error "autograd-hhlo: reduce_window(add) VJP only supports non-overlapping windows with VALID padding"
        else do
            -- For non-overlapping sum-pooling, each output gradient is
            -- broadcast back to its window.
            dx <- broadcastToInputShape bar xShape windowDims strides
            accumulate cmap xVid dx

vjpReduceWindowMax :: BTensor -> BTensor -> TensorType -> [Int64] -> [Int64] -> [[Int64]] -> Int -> [Integer] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpReduceWindowMax bar x xType windowDims strides padding _rank xShape cmap = do
    -- Only non-overlapping windows with VALID padding.
    let isNonOverlapping = and (zipWith (==) windowDims strides)
        isValid = all (all (== 0)) padding
    if not (isNonOverlapping && isValid)
        then error "autograd-hhlo: reduce_window(max) VJP only supports non-overlapping windows with VALID padding"
        else do
            let zeroValType = TensorType [] (ttDType xType)
                -- Compute the reduced output shape for NHWC non-overlapping pooling.
                outShape = map (\(sz, w) -> (sz - fromIntegral w) `div` fromIntegral w + 1) (zip xShape windowDims)
                outType = TensorType outShape (ttDType xType)
            -- Recompute forward max.
            negInf <- bconstant zeroValType (-1.0e30)
            maxVals <- breduceWindowMax x negInf windowDims strides padding outType
            -- Broadcast maxVals back to input shape.
            maxBroadcast <- broadcastToInputShape maxVals xShape windowDims strides
            -- Broadcast bar (gradient) back to input shape.
            barBroadcast <- broadcastToInputShape bar xShape windowDims strides
            -- mask = (input == maxBroadcast)
            mask <- bcompareEQ x maxBroadcast
            -- dx = select(mask, barBroadcast, 0)
            zero <- bconstant xType 0.0
            dx <- bselect mask barBroadcast zero xType
            accumulate cmap (btVid x) dx

-- | Broadcast a reduced tensor back to the original input shape for
-- non-overlapping reduce_window.
-- | Broadcast a reduced tensor back to the original input shape for
-- non-overlapping reduce_window (NHWC with 2 spatial dims).
broadcastToInputShape :: BTensor -> [Integer] -> [Int64] -> [Int64] -> Builder BTensor
broadcastToInputShape reduced xShape windowDims _strides = do
    let reducedShape = ttShape (btType reduced)
        -- reducedShape = [N, outH, outW, C]
        -- Insert size-1 after outH and outW.
        reshapedShape = [reducedShape !! 0, reducedShape !! 1, 1, reducedShape !! 2, 1, reducedShape !! 3]
        -- Broadcast to [N, outH, kh, outW, kw, C]
        broadcastShape = [reducedShape !! 0, reducedShape !! 1, fromIntegral (windowDims !! 1), reducedShape !! 2, fromIntegral (windowDims !! 2), reducedShape !! 3]
        broadcastDims = [0, 1, 2, 3, 4, 5] :: [Int64]
    reshaped <- breshape reduced (TensorType reshapedShape (ttDType (btType reduced)))
    broadcasted <- bbroadcastInDim reshaped broadcastDims (TensorType broadcastShape (ttDType (btType reduced)))
    -- Reshape back to [N, H, W, C]
    breshape broadcasted (TensorType xShape (ttDType (btType reduced)))

findRawIntList :: Text -> [Attribute] -> [Int64]
findRawIntList _ [] = []
findRawIntList name (AttrIntList n v : rest)
    | n == name = v
    | otherwise = findRawIntList name rest
findRawIntList name (AttrRaw raw : rest) =
    let key = name <> " = array<i64:"
    in if key `T.isInfixOf` raw
        then parseIntList raw
        else findRawIntList name rest
findRawIntList name (_ : rest) = findRawIntList name rest

findPadding :: [Attribute] -> [[Int64]]
findPadding [] = []
findPadding (AttrRaw raw : rest) =
    let key = "padding = dense<"
    in if key `T.isInfixOf` raw
        then parsePadding raw
        else findPadding rest
findPadding (_ : rest) = findPadding rest

parsePadding :: Text -> [[Int64]]
parsePadding raw =
    let inner = extractNestedBrackets raw
        pairs = T.splitOn "], [" inner
    in map parsePair pairs
  where
    extractNestedBrackets txt =
        let afterFirstBracket = T.tail $ T.dropWhile (/= '[') txt
        in go afterFirstBracket 1 ""
      where
        go txt' depth acc
            | T.null txt' = acc
            | T.head txt' == ']' && depth == 1 = acc
            | T.head txt' == '[' = go (T.tail txt') (depth + 1) (acc <> "[")
            | T.head txt' == ']' = go (T.tail txt') (depth - 1) (acc <> "]")
            | otherwise = go (T.tail txt') depth (acc <> T.pack [T.head txt'])
    parsePair t =
        let nums = T.splitOn ", " (T.filter (/= '[') (T.filter (/= ']') t))
        in map (read . T.unpack . T.strip) nums

-- ---------------------------------------------------------------------------
-- Convolution VJP rule (NHWC only)
-- ---------------------------------------------------------------------------

vjpConvolution :: Operation -> [Maybe BTensor] -> Map ValueId BTensor -> Builder (Map ValueId BTensor)
vjpConvolution op resultBars cmap = case getResultBar resultBars of
    Just bar -> do
        let input = operandBT op 0
            kernel = operandBT op 1
            inputType = btType input
            kernelType = btType kernel
            attrs = opAttributes op
            kif = fromMaybe (-1) $ lookupAttrInt "kernel_input_feature_dimension" attrs
            kof = fromMaybe (-1) $ lookupAttrInt "kernel_output_feature_dimension" attrs
            stride  = fromMaybe [1,1] $ lookupAttrIntList "window_strides" attrs
            padding = fromMaybe (replicate 4 0) $ lookupAttrIntList "padding" attrs
            lhsDilate = fromMaybe [1,1] $ lookupAttrIntList "lhs_dilation" attrs
            padPairs = [[padding !! 0, padding !! 1], [padding !! 2, padding !! 3]]
        -- Dispatch based on kernel feature dimension order.
        -- Regular conv: kernel_input=2, kernel_output=3
        -- Transpose conv: kernel_input=3, kernel_output=2
        if kif == 2 && kof == 3
            then do
                -- Regular convolution.
                dInput <- convBackwardInput bar kernel kernelType inputType stride padPairs
                cmap' <- accumulate cmap (btVid input) dInput
                let needKernelGrad = btVid kernel < 0 || Map.member (btVid kernel) cmap
                if needKernelGrad
                    then do
                        dKernel <- convBackwardKernel input inputType bar stride padPairs
                        accumulate cmap' (btVid kernel) dKernel
                    else return cmap'
            else if kif == 3 && kof == 2
            then do
                -- Transposed convolution.
                dInput <- transposeConvBackwardInput bar kernel inputType lhsDilate padPairs
                cmap' <- accumulate cmap (btVid input) dInput
                let needKernelGrad = btVid kernel < 0 || Map.member (btVid kernel) cmap
                if needKernelGrad
                    then do
                        dKernel <- transposeConvBackwardKernel input inputType bar lhsDilate padPairs
                        accumulate cmap' (btVid kernel) dKernel
                    else return cmap'
            else error "autograd-hhlo: convolution VJP only supports NHWC canonical attrs"
    Nothing -> return cmap

convBackwardInput :: BTensor -> BTensor -> TensorType -> TensorType -> [Int64] -> [[Int64]] -> Builder BTensor
convBackwardInput bar kernel kernelType inputType stride pad = do
    -- Flip kernel spatially (dims 0 and 1).
    flippedKernel <- breverse kernel [0, 1] kernelType
    -- Backward input uses transposed conv dimension numbers.
    -- Window attributes only apply to spatial dims (first 2 of kernel shape).
    let spatialKernelShape = take 2 (ttShape kernelType)
        spatialReversePad = reversePad pad stride spatialKernelShape
        -- When stride == 1, the backward is a regular conv (lhs_dilate=1).
        -- When stride > 1, the backward is a transpose conv; we must adjust
        -- padding because XLA forward conv uses floor division.
        adjustedPad = if all (== 1) stride
            then spatialReversePad
            else
                let barShape = ttShape (btType bar)
                    inputShape = ttShape inputType
                    spatialBar = take 2 (tail barShape)
                    spatialInput = take 2 (tail inputShape)
                    strideInt = map fromIntegral stride :: [Integer]
                    targetPadTotal = zipWith4 (\inp bar_ stride_ k -> inp - (bar_ - 1) * stride_ + k - 2) (map fromIntegral spatialInput :: [Integer]) (map fromIntegral spatialBar :: [Integer]) strideInt (map fromIntegral spatialKernelShape :: [Integer])
                    actualPadTotal = map (fromIntegral . sum) spatialReversePad :: [Integer]
                    padDiffs = map fromIntegral (zipWith (-) targetPadTotal actualPadTotal) :: [Int64]
                in zipWith (\[l, h] d ->
                    let h' = h + d
                    in if h' >= 0 then [l, h'] else [l + h', 0]) spatialReversePad padDiffs
        windowStr = buildWindowString [1, 1] adjustedPad stride [1, 1]
    bconvolution bar flippedKernel "[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]" windowStr [AttrInt "batch_group_count" 1, AttrInt "feature_group_count" 1] inputType

convBackwardKernel :: BTensor -> TensorType -> BTensor -> [Int64] -> [[Int64]] -> Builder BTensor
convBackwardKernel input inputType bar stride pad = do
    -- Transpose input: [N, H, W, C_in] -> [H, W, C_in, N]
    let inputShape = ttShape inputType
        inputTShape = tail inputShape ++ [head inputShape]
    inputT <- btranspose input [1, 2, 3, 0] (TensorType inputTShape (ttDType inputType))
    -- Transpose bar (dy): [N, outH, outW, C_out] -> [outH, outW, N, C_out]
    let barShape = ttShape (btType bar)
        barTShape = tail barShape ++ [head barShape]
    barT <- btranspose bar [1, 2, 3, 0] (TensorType barTShape (ttDType (btType bar)))
    -- Convolve with adapted dim numbers.  Window uses spatial dims only.
    let spatialKernelShape = take 2 (tail inputShape)
        outType = TensorType (spatialKernelShape ++ [last inputShape, last barShape]) (ttDType inputType)
        windowStr = buildWindowString stride pad [1, 1] [1, 1]
    dk <- bconvolution inputT barT "[0, 1, f, b]x[0, 1, b, f]->[0, 1, i, o]" windowStr [AttrInt "batch_group_count" 1, AttrInt "feature_group_count" 1] outType
    -- Transpose output dims 2 and 3: [kh, kw, C_in, C_out] -> [kh, kw, C_out, C_in]
    btranspose dk [0, 1, 3, 2] outType

-- ---------------------------------------------------------------------------
-- Transpose convolution backward helpers
-- ---------------------------------------------------------------------------

transposeConvBackwardInput :: BTensor -> BTensor -> TensorType -> [Int64] -> [[Int64]] -> Builder BTensor
transposeConvBackwardInput bar kernel inputType lhsDilate pad = do
    -- Backward input: conv(dy, kernel) with stride = lhs_dilate.
    -- Padding must be reversed for the backward regular conv.
    let spatialKernelShape = take 2 (ttShape (btType kernel))
        spatialReversePad = reversePad pad lhsDilate spatialKernelShape
        windowStr = buildWindowString lhsDilate spatialReversePad [1, 1] [1, 1]
    bconvolution bar kernel "[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]" windowStr [AttrInt "batch_group_count" 1, AttrInt "feature_group_count" 1] inputType

transposeConvBackwardKernel :: BTensor -> TensorType -> BTensor -> [Int64] -> [[Int64]] -> Builder BTensor
transposeConvBackwardKernel input inputType bar lhsDilate pad = do
    -- Transpose input: [N, H, W, C_in] -> [H, W, C_in, N]
    let inputShape = ttShape inputType
        inputTShape = tail inputShape ++ [head inputShape]
    inputT <- btranspose input [1, 2, 3, 0] (TensorType inputTShape (ttDType inputType))
    -- Transpose bar: [N, outH, outW, C_out] -> [outH, outW, N, C_out]
    let barShape = ttShape (btType bar)
        barTShape = tail barShape ++ [head barShape]
    barT <- btranspose bar [1, 2, 3, 0] (TensorType barTShape (ttDType (btType bar)))
    -- Convolve with lhs_dilate = forward_lhs_dilate.  Window uses spatial dims only.
    let spatialKernelShape = take 2 (tail inputShape)
        outType = TensorType (spatialKernelShape ++ [last inputShape, last barShape]) (ttDType inputType)
        windowStr = buildWindowString [1, 1] pad lhsDilate [1, 1]
    dk <- bconvolution inputT barT "[0, 1, f, b]x[0, 1, b, f]->[0, 1, i, o]" windowStr [AttrInt "batch_group_count" 1, AttrInt "feature_group_count" 1] outType
    -- Transpose dims 2 and 3: [kh, kw, C_in, C_out] -> [kh, kw, C_out, C_in]
    btranspose dk [0, 1, 3, 2] outType

-- ---------------------------------------------------------------------------
-- Shared attribute parsing helpers
-- ---------------------------------------------------------------------------

parseIntList :: Text -> [Int64]
parseIntList raw =
    let inner = T.takeWhile (/= '>') $ T.dropWhile (/= '<') raw
        withoutLt = T.dropWhile (== '<') inner
        withoutPrefix = if "i64:" `T.isPrefixOf` withoutLt then T.drop 4 withoutLt else withoutLt
        nums = T.splitOn "," withoutPrefix
    in map ((fromIntegral :: Integer -> Int64) . read . T.unpack . T.strip) nums

-- | Parse a plain bracketed int list like @[1, 2]@.
parsePlainIntList :: Text -> [Int64]
parsePlainIntList raw =
    let inner = T.takeWhile (/= ']') $ T.dropWhile (/= '[') raw
        withoutBracket = T.dropWhile (== '[') inner
        nums = T.splitOn "," withoutBracket
    in map (parseNum raw) nums
  where
    parseNum original t =
        let s = T.unpack (T.strip t)
        in case reads s of
            [(n, "")] -> fromIntegral (n :: Integer)
            _ -> error $ "parsePlainIntList: cannot parse '" ++ s ++ "' from raw='" ++ T.unpack original ++ "'"

-- ---------------------------------------------------------------------------
-- Window attribute parsing helpers
-- ---------------------------------------------------------------------------

parseWindowString :: Text -> ([Int64], [[Int64]], [Int64], [Int64])
parseWindowString txt =
    let s = findField "stride" txt
        p = findField "pad" txt
        ld = findField "lhs_dilate" txt
        rd = findField "rhs_dilate" txt
    in ( parsePlainIntList s
       , parsePad p
       , if T.null ld then [1, 1] else parsePlainIntList ld
       , if T.null rd then [1, 1] else parsePlainIntList rd
       )
  where
    -- Find a field value, handling nested brackets for the 'pad' field.
    findField :: Text -> Text -> Text
    findField name t =
        let prefix = name <> " = "
        in case T.breakOn prefix t of
            (_, rest) | T.null rest -> ""
            (_, rest') ->
                let rest = T.drop (T.length prefix) rest'
                in takeValue rest

    -- Take the value, respecting bracket nesting.
    takeValue :: Text -> Text
    takeValue t = takeValueGo t (0 :: Int) ""
      where
        takeValueGo :: Text -> Int -> Text -> Text
        takeValueGo txt bracketDepth acc
            | T.null txt = acc
            | T.head txt == '}' && bracketDepth == 0 = acc
            | T.head txt == ',' && bracketDepth == 0 = acc
            | T.head txt == '[' = takeValueGo (T.tail txt) (bracketDepth + 1) (acc <> "[")
            | T.head txt == ']' = takeValueGo (T.tail txt) (bracketDepth - 1) (acc <> "]")
            | otherwise = takeValueGo (T.tail txt) bracketDepth (acc <> T.pack [T.head txt])

    parsePad :: Text -> [[Int64]]
    parsePad t
        | T.null t  = [[0, 0], [0, 0]]
        | otherwise =
            let -- Extract content between outermost [[ and ]]
                inner = extractNestedBrackets t
                pairs = T.splitOn "], [" inner
            in map parsePair pairs
      where
        extractNestedBrackets txt =
            let afterFirstBracket = T.tail $ T.dropWhile (/= '[') txt
            in go afterFirstBracket 1 ""
          where
            go txt' depth acc
                | T.null txt' = acc
                | T.head txt' == ']' && depth == 1 = acc
                | T.head txt' == '[' = go (T.tail txt') (depth + 1) (acc <> "[")
                | T.head txt' == ']' = go (T.tail txt') (depth - 1) (acc <> "]")
                | otherwise = go (T.tail txt') depth (acc <> T.pack [T.head txt'])

    parsePair :: Text -> [Int64]
    parsePair t =
        let nums = T.splitOn ", " (T.filter (/= '[') (T.filter (/= ']') t))
        in map parseNum nums
      where
        parseNum t' =
            let s = T.unpack (T.strip t')
            in case reads s of
                [(n, "")] -> fromIntegral (n :: Integer)
                _ -> error $ "parsePair: cannot parse '" ++ s ++ "' from raw='" ++ T.unpack t ++ "'"

reversePad :: [[Int64]] -> [Int64] -> [Integer] -> [[Int64]]
reversePad pad _stride kernelShape =
    zipWith go pad (map fromIntegral kernelShape)
  where
    go [l, h] k = [fromIntegral (k :: Integer) - 1 - h, fromIntegral k - 1 - l]
    go _      _ = error "reversePad: invalid padding pair"

buildWindowString :: [Int64] -> [[Int64]] -> [Int64] -> [Int64] -> Text
buildWindowString stride pad lhsDilate rhsDilate =
    "{stride = " <> showList' stride
    <> ", pad = " <> showPad pad
    <> ", lhs_dilate = " <> showList' lhsDilate
    <> ", rhs_dilate = " <> showList' rhsDilate <> "}"
  where
    showList' xs = "[" <> T.intercalate ", " (map (T.pack . show) xs) <> "]"
    showPad ps = "[" <> T.intercalate ", " (map showPair ps) <> "]"
    showPair [l, h] = "[" <> T.pack (show l) <> ", " <> T.pack (show h) <> "]"
    showPair _      = error "buildWindowString: invalid padding pair"

lookupAttrInt :: Text -> [Attribute] -> Maybe Int64
lookupAttrInt name = listToMaybe . foldr f []
  where
    f (AttrInt n v) acc | n == name = v : acc
    f _ acc = acc

lookupAttrIntList :: Text -> [Attribute] -> Maybe [Int64]
lookupAttrIntList name = listToMaybe . foldr f []
  where
    f (AttrIntList n v) acc | n == name = v : acc
    f _ acc = acc

lookupAttrString :: Text -> [Attribute] -> Text
lookupAttrString name = foldr f ""
  where
    f (AttrString n s) acc | n == name = s <> acc
    f _ acc = acc
