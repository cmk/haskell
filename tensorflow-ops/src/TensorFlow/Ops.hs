-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CNDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- | This module contains definitions for some built-in TensorFlow operations.
--
-- Note that certain, "stateful" ops like 'variable' and 'assign' return a
-- 'Build' action (e.g., @Build (Tensor Ref a)@ instead of a pure value; the
-- returned 'Tensor's are always rendered in the current 'Build' context.  This
-- approach helps us avoid problems with inlining or common subexpression
-- elimination, by writing
--
-- > do
-- >     v <- variable []
-- >     w <- assign v 3
-- >     render $ w * w
--
-- instead of
--
-- > let
-- >    v = variable []
-- >    w = assign v 3
-- > in w * w
--
-- since the latter could be reasonably transformed by the compiler into (or
-- vice versa)
--
-- > let
-- >    v = variable []
-- >    w = assign v 3
-- >    w' = assign v 3
-- > in w * w'
--
-- Ops should return a 'Build' action if their original 'OpDef' marks them as
-- stateful, or if they take any Refs as input.  (This mirrors the rules that
-- TensorFlow uses to avoid common subexpression elimination.)
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

{-# LANGUAGE TypeOperators #-}

{-# LANGUAGE ViewPatterns #-}

module TensorFlow.Ops
    ( C.add
    , C.add'
    , C.abs
    , C.abs'
    , C.addN
    , C.addN'
    , C.argMax
    , C.argMax'
    , C.assign
    , C.assign'
    , C.broadcastGradientArgs
    , C.broadcastGradientArgs'
    , C.cast
    , C.cast'
    , C.concat
    , C.concat'
    , constant
    , constant'
    , C.equal
    , C.equal'
    , expandDims
    , expandDims'
    , initializedVariable
    , initializedVariable'
    , zeroInitializedVariable
    , zeroInitializedVariable'
    , C.fill
    , C.fill'
    , C.identity
    , C.identity'
    , C.matMul
    , C.matMul'
    , matTranspose
    , matTranspose'
    , C.mean
    , C.mean'
    , C.mul
    , C.mul'
    , C.neg
    , C.neg'
    , C.oneHot
    , C.oneHot'
    , C.pack
    , C.pack'
    , placeholder
    , placeholder'
    , C.range
    , C.range'
    , reducedShape
    , reduceMean
    , reduceMean'
    , C.relu
    , C.relu'
    , C.reluGrad
    , C.reluGrad'
    , C.reshape
    , C.reshape'
    , restore
    , restoreFromName
    , save
    , scalar
    , scalar'
    , shape
    , shape'
    , C.sign
    , C.sign'
    , C.size
    , C.size'
    , C.softmax
    , C.softmax'
    , C.softmaxCrossEntropyWithLogits
    , C.softmaxCrossEntropyWithLogits'
    , C.sparseToDense
    , C.sparseToDense'
    , C.sub
    , C.sub'
    , C.sum
    , C.sum'
    , reduceSum
    , reduceSum'
    , C.tanh
    , C.tanhGrad
    , C.transpose
    , C.transpose'
    , truncatedNormal
    , truncatedNormal'
    , C.variable
    , C.variable'
    , vector
    , vector'
    , zeros
    , C.zerosLike
    , C.zerosLike'
    , scalarize

    , squareD
    , mulD
    , addD
    , subD
    , matMulD
    , myMeanD
    ) where

import Data.ByteString (ByteString)
import Data.Complex (Complex)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Word (Word8, Word16)
import Prelude hiding (abs, sum, concat)
import Data.ProtoLens (def)
import Data.Text.Encoding (encodeUtf8)
import Lens.Family2 (Lens', (.~), (&),(^.))
import Lens.Family2.Stock (at, intAt)
import Lens.Family2.Unchecked (lens, iso)

import Text.Printf (printf)
import Proto.Tensorflow.Core.Framework.Tensor (TensorProto)
import Proto.Tensorflow.Core.Framework.Tensor_Fields -- watch out for this
    ( dtype
    , tensorShape
    )
import Proto.Tensorflow.Core.Framework.Tensor_Fields
    ( dtype
    , tensorShape
    )
import qualified Proto.Tensorflow.Core.Framework.TensorShape_Fields -- watch out for this
  as TensorShape
import TensorFlow.Build
import TensorFlow.BuildOp
import TensorFlow.ControlFlow (group)
import TensorFlow.Tensor
import TensorFlow.Types

import TensorFlow.Output ( NodeName(..))
import qualified TensorFlow.GenOps.Core as C

import Data.Text (Text)
import qualified Prelude (abs)
import Data.Maybe
--import Proto.Tensorflow.Core.Framework.NodeDef ( attr, input, op, name)
--import Proto.Tensorflow.Core.Framework.NodeDef (NodeDef)
import Proto.Tensorflow.Core.Framework.NodeDef_Fields (attr, op, input, name)
import Proto.Tensorflow.Core.Framework.NodeDef (NodeDef)


import Numeric.Backprop
import Numeric.Backprop.Class
-- TODO: Look into hs-boot refactoring to allow mutually recursive imports.
-- | Must be defined as an orphan because of the dependency order between Ops
-- and Tensor.
--
-- The indirect constraint "v ~ Value" helps disambiguate types, for example in
-- "neg 1 :: Tensor Value Float", it helps find the type of the subexpression
-- "1".
instance ( TensorType a
         , Num a
         , v ~ Build
         , OneOf '[ Double, Float, Int32, Int64
                  , Complex Float, Complex Double] a) => Num (Tensor v a) where
    (+) = C.add
    (*) = C.mul
    (-) = C.sub
    abs = C.abs
    fromInteger = scalar . fromInteger
    signum = C.sign
    negate = C.neg

instance ( TensorType a
         , Num a
         , v ~ Build
         , OneOf '[ Double, Float, Int32, Int64
                  , Complex Float, Complex Double] a) => Backprop (Tensor v a) where
    zero = zeroNum
    add  = addNum
    one  = oneNum

type T a = Tensor Build a

non :: Eq a => a -> Lens' (Maybe a) a
non a = anon a (a==)

-- Copy of http://hackage.haskell.org/package/lens-3.9.0.2/docs/Control-Lens-Iso.html#v%3anon
anon :: a -> (a -> Bool) -> Lens' (Maybe a) a
anon a p = iso (fromMaybe a) go where
  go b | p b       = Nothing
       | otherwise = Just b


lookupAttr :: Attribute a1 => NodeDef -> Text -> a1
lookupAttr nodeDef attrName = nodeDef ^. attr . at attrName . non def . attrLens

-- TODO: Double check all these /= restrictions after bumping backprop version
squareD
    :: (TensorType a, Num a, a /= Int8, a /= Int16,
        a /= Word8, a /= Word16, a /= ByteString, a /= Bool, Reifies s W)

    => BVar s (T a)
    -> BVar s (T a)
squareD = liftOp1 . op1 $ \x ->
  ( x * x, \dzdy -> dzdy * 2 * x)


mulD 
    :: (TensorType a, Num a, a /= Int8, a /= Int16,
        a /= Word8, a /= Word16, a /= ByteString, a /= Bool, Reifies s W)
    => BVar s (T a)
    -> BVar s (T a)
    -> BVar s (T a)
mulD = liftOp2 . op2 $ \x1 x2 ->
  (x1 * x2, \dzdy -> (dzdy * x2, x1 * dzdy))

addD
    :: (TensorType a, Num a, a /= Bool, a /= Int8, a /= Int16, a /= Word8,
        a /= Word16, a /= ByteString, Reifies s W)
    => BVar s (T a)
    -> BVar s (T a)
    -> BVar s (T a)
addD = liftOp2 . op2 $ \x1 x2 ->
  (x1 + x2, \dzdy -> (dzdy, dzdy))

subD
    :: (TensorType a, Num a, a /= Bool, a /= Int8, a /= Int16, a /= Word8,
        a /= Word16, a /= ByteString, Reifies s W)
    => BVar s (T a)
    -> BVar s (T a)
    -> BVar s (T a)
subD = liftOp2 . op2 $ \x1 x2 ->
  (x1 - x2, \dzdy -> (dzdy, negate dzdy))

matMulD :: (TensorType a, Num a, a /= Int8, a /= Int16, a /= Int64, a /= Word8,
            a /= Word16, a /= ByteString, a /= Bool, Reifies s W )
    => BVar s (T a)
    -> BVar s (T a)
    -> BVar s (T a)
matMulD = 
  let transAttrs a b = (opAttr "transpose_a" .~ a) . (opAttr "transpose_b" .~ b)

  in liftOp2 . op2 $ \x y ->
  (C.matMul x y, \dz -> (C.matMul' (transAttrs False True) dz y, C.matMul' (transAttrs True False) x dz))

myMeanD
  :: (TensorType t, Num t, t /= Bool, t /= Int8, t /= Int16, t /= Word8, t /= Word16, t /= ByteString, Reifies s W) 
    => BVar s (T t)
    -> BVar s (T t)
myMeanD = liftOp1 . op1 $ \x -> (myMean x, id) -- extremely suspect

myMean x = C.mean' id x allAxes
  where allAxes = C.range 0 (C.rank x :: Tensor Build Int32) 1


{-

liftOp1 . op $ \x1 x2 ->
  (C.mean x1 x2, \dzdy -> (dzdy, negate dzdy))

mean = liftOp1 . op1 $ \x -> (H.mean x, H.konst . (/ H.norm_0 x))
  



    [Just $ dz `CoreOps.div` CoreOps.cast factor, Nothing]
  where
    [Just dz, Nothing] = opGrad "Sum" u v w
    inputShape = shape (x :: Tensor Build a)
    outputShape = shape (dz :: Tensor Build a)
    -- TODO(fmayle): Add fast path when shape is known.
    inputSize = CoreOps.prod inputShape $ rangeOfRank inputShape
    outputSize = CoreOps.prod outputShape $ rangeOfRank outputShape
    factor = safeShapeDiv inputSize outputSize


matMul
  :: (TF.TensorType t, t TF./= Bool,
      t TF./= Data.ByteString.Internal.ByteString,
      t TF./= GHC.Word.Word8, t TF./= Int64, t TF./= Int16,
      t TF./= Int8) =>
     Tensor v'1 t -> Tensor v'2 t -> Tensor Build t


opGrad "MatMul" nodeDef [toT -> x, toT -> y] [dz] =
    let transposeA = lookupAttr nodeDef "transpose_a"
        transposeB = lookupAttr nodeDef "transpose_b"
        transAttrs a b =
            (opAttr "transpose_a" .~ a) . (opAttr "transpose_b" .~ b)
    in case (transposeA, transposeB) of
       (False, False) ->
           [ Just $ matMul' (transAttrs False True) dz y
           , Just $ matMul' (transAttrs True False) x dz]
       (False, True) ->
           [ Just $ matMul dz y
           , Just $ matMul' (transAttrs True False) dz x]
       (True, False) ->
           [ Just $ matMul' (transAttrs False True) y dz
           , Just $ matMul x dz]
       (True, True) ->
           [ Just $ matMul' (transAttrs True True) y dz
           , Just $ matMul' (transAttrs True True) dz x]

mul :: ( KnownNat m
       , KnownNat k
       , KnownNat n
       , H.Domain field vec mat
       , Backprop (mat m k)
       , Backprop (mat k n)
       , HU.Transposable (mat m k) (mat k m)
       , HU.Transposable (mat k n) (mat n k)
       , Reifies s W
       )
    => BVar s (mat m k)
    -> BVar s (mat k n)
    -> BVar s (mat m n)
mul = liftOp2 . op2 $ \x y ->
    ( x `H.mul` y
    , \d -> (d `H.mul` H.tr y, H.tr x `H.mul` d)
    )


maxD
  :: (TensorType i, TensorType t, Num t, i ~ Int32, t /= ByteString,
      t /= Int8, t /= Int16, t /= Word8, t /= Word16, t /= Bool, Reifies s W)
    => BVar s (T t)
    -> BVar s (T i)
    -> BVar s (T t)
maxD = liftOp2 . op2 $ f 
  where 
    --f :: T t -> T Int32 -> (T t, T t -> (T t, T Int32)) 
    f x i = (y, \dz -> (g dz, 0))
      where
        y = C.max x i
        y' = C.reshape y outputShapeKeptDims
        g dz = indicators `C.div` numSelected * (C.reshape dz outputShapeKeptDims)
        indicators = C.cast $ C.equal y' x
        numSelected = C.reshape (C.sum indicators i) outputShapeKeptDims
        outputShapeKeptDims = reducedShape sx (i :: Tensor Build Int32)
        sx = shape (x :: Tensor Build a)
-}


matTranspose :: TensorType a => Tensor e a -> Tensor Build a
matTranspose = matTranspose' id

matTranspose' :: TensorType a => OpParams -> Tensor v a -> Tensor Build a
matTranspose' params = flip (C.transpose' params) (vector [1, 0 :: Int32])

placeholder :: (MonadBuild m, TensorType a) => Shape -> m (Tensor Value a)
placeholder = placeholder' id

placeholder' :: forall m a . (MonadBuild m, TensorType a)
             => OpParams -> Shape -> m (Tensor Value a)
placeholder' params pShape
    -- Note: we don't use C.placeholder' since that op isn't stateful,
    -- and thus would be CSE'd.
    = build $ buildOp [] $ opDef "Placeholder"
                & opAttr "dtype" .~ tensorType (undefined :: a)
                & opAttr "shape" .~ pShape
                & params

-- | Creates a variable initialized to the given value.
-- Initialization happens next time session runs.
initializedVariable :: (MonadBuild m, TensorType a)
                    => Tensor v a -> m (Tensor Ref a)
initializedVariable = initializedVariable' id

initializedVariable' :: (MonadBuild m, TensorType a)
                    => OpParams -> Tensor v a -> m (Tensor Ref a)
initializedVariable' params initializer = do
    v <- C.variable' params []  -- The shape is not known initially.
    i <- C.assign' (opAttr "validate_shape" .~ False) v
                            initializer
    addInitializer =<< group i
    return v

-- | Creates a zero-initialized variable with the given shape.
zeroInitializedVariable
  :: (MonadBuild m, TensorType a, Num a) =>
     TensorFlow.Types.Shape -> m (Tensor TensorFlow.Tensor.Ref a)
zeroInitializedVariable = zeroInitializedVariable' id

zeroInitializedVariable'
  :: (MonadBuild m, TensorType a, Num a) =>
     OpParams -> TensorFlow.Types.Shape -> m (Tensor TensorFlow.Tensor.Ref a)
zeroInitializedVariable' params = initializedVariable' params . zeros

-- TODO: Support heterogeneous list of tensors.
save :: forall a m v . (Rendered (Tensor v), MonadBuild m, TensorType a)
        => ByteString    -- ^ File path.
        -> [Tensor v a]  -- ^ Tensors to save.
        -> m ControlNode
save path xs = build $ do
    let toByteStringTensor = scalar . encodeUtf8 . encodeOutput . renderedOutput
    let names = fmap toByteStringTensor xs
    let types = replicate (length xs) (tensorType (undefined :: a))
    names' <- buildInputs $ C.pack names
    xs' <- buildInputs xs
    path' <- buildInputs $ scalar path
    buildOp [] $ opDef "Save"
                    & opAttr "T" .~ types
                    & opInputs .~ (path' ++ names' ++ xs')

-- | Restore a tensor's value from a checkpoint file.
--
-- This version allows restoring from a checkpoint file that uses a different
-- tensor name than the variable.
restoreFromName :: forall a m . (MonadBuild m, TensorType a)
                => ByteString    -- ^ File path.
                -> ByteString    -- ^ Tensor name override.
                -> Tensor Ref a  -- ^ Tensor to restore.
                -> m ControlNode
restoreFromName path name x = build $ do
    path' <- buildInputs $ scalar path
    name' <- buildInputs $ scalar name
    restoreOp <- buildOp [] $ opDef "Restore"
                               & opAttr "dt" .~ tensorType (undefined :: a)
                               & opInputs .~ (path' ++ name')
    group =<< C.assign x (restoreOp :: Tensor Value a)

-- | Restore a tensor's value from a checkpoint file.
restore :: forall a m . (MonadBuild m, TensorType a)
        => ByteString    -- ^ File path.
        -> Tensor Ref a  -- ^ Tensor to restore.
        -> m ControlNode
restore path x = restoreFromName path name x
  where
    name = encodeUtf8 $ encodeOutput $ renderedOutput x

-- | Create a constant tensor.
--
-- The values should be in row major order, e.g.,
--
--   element 0:   index (0, ..., 0)
--   element 1:   index (0, ..., 1)
--   ...
constant :: TensorType a => Shape -> [a] -> Tensor Build a
constant = constant' id

constant' :: forall a . TensorType a => OpParams -> Shape -> [a] -> Tensor Build a
constant' params (Shape cShape) values
    | invalidLength = error invalidLengthMsg
    | otherwise = C.const' (params . (opAttr "value" .~ typedNode))
  where
    invalidLength = product cShape /= fromIntegral (length values)
    invalidLengthMsg = printf "invalid tensor length: expected %d got %d"
                              (product cShape)
                              (length values)
    typedNode :: TensorProto
    typedNode = def
                & dtype .~ tensorType (undefined :: a)
                & tensorShape.TensorShape.dim .~
                      [def & TensorShape.size .~ x | x <- cShape]
                & tensorVal .~ values

-- | Reshape a N-D tensor down to a scalar.
--
-- See `TensorFlow.GenOps.Core.reshape`.
scalarize :: TensorType a => Tensor v a -> Tensor Build a
scalarize t = C.reshape t (vector scalarShape)
    where
        scalarShape = [] :: [Int32]

-- | Sum a tensor down to a scalar
-- Seee `TensorFlow.GenOps.Core.sum`
reduceSum :: (OneOf '[ Double, Float, Int32, Int64
                     , Complex Float, Complex Double] a) =>
             Tensor v a -> Tensor Build a
reduceSum x = C.sum x allAxes
  where allAxes = C.range 0 (C.rank x :: Tensor Build Int32) 1

reduceSum' :: (OneOf '[ Double, Float, Int32, Int64
                      , Complex Float, Complex Double] a) =>
              OpParams -> Tensor v a -> Tensor Build a
reduceSum' params x = C.sum' params x allAxes
  where allAxes = C.range 0 (C.rank x :: Tensor Build Int32) 1

-- | Computes the mean of elements across dimensions of a tensor.
-- See `TensorFlow.GenOps.Core.mean`
reduceMean
  :: ( TensorType a
     , OneOf '[ Double, Float, Complex Float, Complex Double] a
     )
  => Tensor v a -> Tensor Build a
reduceMean = reduceMean' id

reduceMean'
  :: ( TensorType a
     , OneOf '[ Double, Float, Complex Float, Complex Double] a
     )
  => OpParams -> Tensor v a -> Tensor Build a
reduceMean' params x = C.mean' params x allAxes
  where allAxes = C.range 0 (C.rank x :: Tensor Build Int32) 1

-- | Create a constant vector.
vector :: TensorType a => [a] -> Tensor Build a
vector = vector' id

vector' :: TensorType a => OpParams -> [a] -> Tensor Build a
vector' params xs = constant' params [fromIntegral $ length xs] xs

-- | Create a constant scalar.
scalar :: TensorType a => a -> Tensor Build a
scalar = scalar' id

scalar' :: TensorType a => OpParams -> a -> Tensor Build a
scalar' params x = constant' params [] [x]

-- | Random tensor from the unit normal distribution with bounded values.
--
-- This is a type-restricted version of 'TensorFlow.GenOps.Core.truncatedNormal'.
truncatedNormal :: (MonadBuild m, OneOf '[Word16, Double, Float] a)
                => Tensor v Int64  -- ^ Shape.
                -> m (Tensor Value a)
truncatedNormal = C.truncatedNormal

truncatedNormal' :: (MonadBuild m, OneOf '[Word16, Double, Float] a)
                => OpParams -> Tensor v Int64  -- ^ Shape.
                -> m (Tensor Value a)
truncatedNormal' = C.truncatedNormal'

zeros :: forall a . (Num a, TensorType a) => Shape -> Tensor Build a
zeros (Shape s) = C.fill (vector s) (scalar 0)

shape :: TensorType t => Tensor v t -> Tensor Build Int32
shape = C.shape

shape' :: TensorType t => OpParams -> Tensor v t -> Tensor Build Int32
shape' = C.shape'

expandDims :: TensorType t => Tensor v1 t -> Tensor v2 Int32 -> Tensor Build t
expandDims = C.expandDims

expandDims' :: TensorType t => OpParams -> Tensor v1 t -> Tensor v2 Int32 -> Tensor Build t
expandDims' = C.expandDims'

-- | Helper function for reduction ops (translation of math_ops.reduced_shape).
reducedShape :: (OneOf '[ Int32, Int64 ] t1, OneOf '[ Int32, Int64 ] t2) =>
                Tensor v1 t1 -> Tensor v2 t2 -> Tensor Build Int32
reducedShape inputShape axes =
    let inputShape32 = toInt32 inputShape         -- [2, 3, 5, 7]
        axes32 = toInt32 axes                     -- [1, 2]
        toInt32 x = C.cast x :: Tensor Build Int32
        inputRank = C.size inputShape32     -- 4
        axesMod = (axes32 + inputRank) `C.mod` inputRank
        axesShape = shape axesMod                 -- [2]
    in C.dynamicStitch                      -- [2, 1, 1, 7]
         [C.range 0 inputRank 1,            -- [0, 1, 2, 3]
           axesMod]                               -- [1, 2]
         [inputShape32,                           -- [2, 3, 5, 7]
           C.fill axesShape 1]              -- [1, 1]
