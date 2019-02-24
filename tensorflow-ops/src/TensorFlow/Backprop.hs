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

module TensorFlow.Backprop where

import qualified Data.Vector as V

import qualified TensorFlow.Ops as TF hiding (initializedVariable)
import TensorFlow.Ops

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as C

import Numeric.Backprop
import Control.Monad.IO.Class (liftIO)
import Data.Functor.Identity (Identity(..))
import Lens.Family2 ((^.), (.~), (^..))

import Control.Monad (replicateM_)   

import TensorFlow.Build
    ( Build
    , BuildT
    , asGraphDef
    , evalBuildT
    , flushNodeBuffer
    , withDevice
    , withNameScope
    , opName
    )
import TensorFlow.Types (unScalar)
import TensorFlow.Output (Device(..))
import TensorFlow.Tensor
import TensorFlow.Session
    ( run
    , runSession
    , run_
    )

import Data.Int (Int8, Int16, Int32, Int64)

import TensorFlow.Gradient (gradients)
import TensorFlow.ControlFlow
import Data.ProtoLens.TextFormat (showMessage)
--import Proto.Tensorflow.Core.Framework.Graph (node)
--import Proto.Tensorflow.Core.Framework.NodeDef (op, input, name)


import Proto.Tensorflow.Core.Framework.Graph_Fields (node)
import Proto.Tensorflow.Core.Framework.NodeDef_Fields (op, input, name)

import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Variable as TF

eval :: TF.Fetchable t a => t -> IO a
eval = TF.runSession . TF.run


minusP :: Tensor Build Float -> (Tensor Build Float, Tensor Build Float)
       -> (Tensor Build Float, Tensor Build Float) 
       -> (Tensor Build Float, Tensor Build Float)
minusP eps (a,b) (c,d) = (a-eps*c, b-eps*d)


randomParam :: TF.Shape -> TF.Session (TF.Tensor TF.Value Float)
randomParam (TF.Shape shape) = TF.truncatedNormal (TF.vector shape)

doN 0 f x = f x
doN n f x = let x' = f x in doN (n-1) f x'

--testMatrix :: IO ()
testMatrix = do
  let ones = [1, 1, 1, 1] :: [Float]
      matx = TF.constant [2, 2] ones

      eps = TF.scalar (0.1 :: Float)
      f x y = (constVar matx) `subD` (x `matMulD` y)
      loss x y = myMeanD $ squareD $ f x y

      trainStep (x,y) = minusP eps (x,y) $ gradBP2 loss x y --TODO use control nodes
      u = TF.constant [2, 1] [0.9, 1.1]
      v = TF.constant [1, 2] [1.5, 1.1]
      g = doN 7 trainStep (u,v)
      

  let graphDef = TF.asGraphDef $ sequence $ render <$> toList g
      ops = graphDef ^.. node . traverse . name
      inps = graphDef ^.. node . traverse . input 
  l <- traverse print $ zip ops inps
  print $ length l

  (u', v') <- eval g 
  print $ V.toList (u' :: V.Vector Float)
  print $ V.toList (v' :: V.Vector Float)
  print $ ((*) <$> u' <*> v')
  return ()

type T a = Tensor Build a

gradBP2' 
  :: (forall s. Reifies s W => BVar s (T Float) -> BVar s (T Float) -> BVar s (T Float))
  -> [TF.Variable Float] -> Build [Tensor Value Float]
gradBP2' f [a,b] = sequence $ fmap render [c,d]  where (c, d) = gradBP2 f (TF.readValue a) (TF.readValue b)
-- BVar s (T a)
-- fitMatrix :: Test
fitMatrix = TF.runSession $ do
  u <- TF.initializedVariable =<< randomParam [2, 1]
  v <- TF.initializedVariable =<< randomParam [1, 2]
  let ones = [1, 1, 1, 1] :: [Float]
      matx = TF.constant [2, 2] ones

      --f x y = (constVar matx) `subD` (x `matMulD` y)
      --loss x y = myMeanD $ squareD $ f x y
      --gradBP2' loss

      diff = matx `C.sub` (TF.readValue u `C.matMul` TF.readValue v)
      loss = TF.reduceMean $ C.square diff

      --f :: [Tensor Value TVal] -> Build ControlNode
      --f params = TF.gradientDescent 0.01 params
      --minimize :: Tensor Build TVal	-> [Variable TVal]	-> Build ControlNode
      minimize loss params = gradients loss params >>= TF.gradientDescent 0.01 params

      
  trainStep <- minimize loss [u, v]
  replicateM_ 1000 (TF.run trainStep)
{-
  let graphDef = TF.asGraphDef $ sequence $ render <$> toList trainStep

  let ops = graphDef ^.. node . traverse . name
      inps = graphDef ^.. node . traverse . input 
  l <- liftIO $ traverse print $ zip ops inps
  liftIO $ print $ length l
-}
  (u' :: V.Vector Float,v' :: V.Vector Float) <- TF.run (TF.readValue u, TF.readValue v)
  -- ones = u * v
  liftIO $ print ((*) <$> u' <*> v')

{-

modify :: Storable a => (forall s. MVector s a -> ST s ()) -> Vector a -> Vector a

read :: (PrimMonad m, Storable a) => MVector (PrimState m) a -> Int -> m a 

write :: (PrimMonad m, Storable a) => MVector (PrimState m) a -> Int -> a -> m () 

thaw :: Vector e -> ST s (MVector s e)

freeze :: MVector s e -> ST s (Vector e)

-}


{-

thaw :: (Elt e, PrimMonad m) => Tensor d e -> m (MTensor (PrimState m) d e)

freeze :: (Elt e, PrimMonad m) => MTensor (PrimState m) d e -> m (Tensor d e)


newtype MTensor s d e = MTensor { unMTensor :: MVector s e } -- vector 
newtype MTensor s d e = MTensor { unMTensor :: Variable e } -- tensorflow
 
readValue :: TensorType a => Variable a -> Tensor Build a

data Variable a = Variable
    { variableHandle   :: Tensor Value ResourceHandle
    , initializedValue :: Maybe (Tensor Value a)
      -- ^ The initial value of a 'Variable' created with 'initializedVariable'.
    }


render :: Tensor Build a -> Build (Tensor Value a)
render (Tensor t) = Tensor . Value <$> build t

renderValue :: MonadBuild m => Tensor v a -> m (Tensor Value a)
renderValue (Tensor o) = render $ Tensor $ toBuild o

-- TODO: better name.
expr :: TensorKind v => Tensor v a -> Tensor Build a
expr (Tensor o) = Tensor $ toBuild o

-- TODO : unify gradBP2' and gradients 
gradBP2'  :: (Backprop a, Backprop b, Backprop c) =>
          (forall s. Reifies s W => BVar s a -> BVar s b -> BVar s c)
          -> a -> b -> (a, b)


-- TODO : ok to go : Tensor Value TVal -> Tensor Build TVal -> Tensor Value TVal ??
-- renderValue . expr == id ??
gradients :: PrimMonad m => Tensor d TVal -> [MTensor (PrimState m) d TVal] -> m [Tensor d TVal]

gradients :: Tensor Build TVal -> [Variable TVal] -> Build [Tensor Value TVal]


minimize :: Tensor Build TVal	-> [Variable TVal]	-> Build ControlNode
minimize loss params = TF.gradients loss params >>= f
  where  f :: [Tensor Value TVal] -> Build ControlNode
         f = gradientDescent 0.01 params


-- | Perform one step of the gradient descent algorithm.
gradientDescent :: TVal -> [Variable TVal] -> [Tensor Value TVal] -> Build ControlNode
gradientDescent learningRate params grads = TF.withNameScope "gradientDescent" $ do

  let applyGrad param grad = TF.assignAdd param (TF.scalar (-learningRate) `TF.mul` grad) -- this in ST
  TF.group =<< zipWithM applyGrad params grads


-- Var
assignAdd :: (MonadBuild m, TensorType a) => Variable a -> Tensor v a -> m ControlNode

assignAddVariableOp' :: forall v'1 v'2 dtype m' . (MonadBuild m',
                                                   TensorType dtype) =>
                        OpParams ->
                        Tensor v'1 ResourceHandle -- ^ __resource__
                        -> Tensor v'2 dtype -- ^ __value__
                        -> m' (ControlNode)
assignAddVariableOp' op'options resource value | eqLengthGuard [] =
    build $ do
        op'inputs <- fmap Prelude.concat $ Prelude.sequence [buildInputs resource,
                                                             buildInputs value]
        buildOp [] (opDef "AssignAddVariableOp"
                    & opAttr "dtype" .~ tensorType (undefined :: dtype)
                    & op'options & opInputs .~ op'inputs)

assignAdd' :: forall v'2 t m' . (MonadBuild m',
                                 OneOf '[(Data.Complex.Complex Double),
                                         (Data.Complex.Complex Float),
                                         Data.Int.Int16, Data.Int.Int32,
                                         Data.Int.Int64, Data.Int.Int8,
                                         Data.Word.Word16, Data.Word.Word32,
                                         Data.Word.Word64, Data.Word.Word8,
                                         Double, Float] t) => OpParams ->
              Tensor Ref t -- ^ __ref__
              -> Tensor v'2 t -- ^ __value__
              -> m' (Tensor Ref t) -- ^ __output_ref__
assignAdd' op'options ref value | eqLengthGuard [] =
    build $ do
        op'inputs <- fmap Prelude.concat $ Prelude.sequence [buildInputs ref,
                                                             buildInputs value]
        buildOp [] (opDef "AssignAdd"
                    & opAttr "T" .~ tensorType (undefined :: t)
                    & op'options & opInputs .~ op'inputs)


- grok tf Variable, how is it like ST?
- 
-
-}
{-
squaredErrorGrad
    :: (Num p, Num b)
    => Model p a b      -- ^ Model
    -> a                -- ^ Observed input
    -> b                -- ^ Observed output
    -> p                -- ^ Parameter guess
    -> p                -- ^ Gradient
squaredErrorGrad f x targ = gradBP $ \p ->
    (f p (auto x) - auto targ) ^ 2

trainModel
    :: (Fractional p, Num p, Num b)
    => Model p a b      -- ^ model to train
    -> p                -- ^ initial parameter guess
    -> [(a,b)]          -- ^ list of observations
    -> p                -- ^ updated parameter guess
trainModel f = foldl' $ \p (x,y) -> p - 0.1 * squaredErrorGrad f x y p
-}


-- inserting this before eval results in an exception
-- breaky :: Tensor Build a -> Tensor Value a
-- breaky = runIdentity . evalBuildT . render
-- *** Exception: TensorFlowException TF_INVALID_ARGUMENT "Session was not created with a graph before Run()!"
--
testBasic :: IO Bool
testBasic = do
  x <- eval $ gradBP squareD (TF.vector [1, 2 :: Float])
  --print $ V.toList (x :: V.Vector Float)
  return $ V.fromList [2, 4 :: Float] == x


testDiamond = do
  let f x = let y = x `mulD` x
                z = y `mulD` y
            in z
      f' x = squareD $ squareD x
      --g :: Tensor Build Float
      g = gradBP f' (TF.vector [1, 2 :: Float])
  x :: V.Vector Float <- eval $ g  --print $ V.toList (x :: V.Vector Float)
  let graphDef = TF.asGraphDef $ render g
  --print $ graphDef

  let ops = graphDef ^.. node . traverse . name
      --expected = ["Add","Mul","Add","Mul","Const","Add","Const","Const"]
      inps = graphDef ^.. node . traverse . input 
  traverse print $ zip ops inps
  --return $ V.fromList [4, 32 :: Float] == x

toList :: (Tensor Build Float, Tensor Build Float) -> [Tensor Build Float]
toList (a,b) = [a,b]


testGradientSimple = do
  let f x b = (x `mulD` x) `addD` b
      g = gradBP2 f (TF.scalar (3 :: Float)) (TF.scalar (4 :: Float))
  (dx, db) <- eval g
    -- Assert that the gradients are right.
  let graphDef = TF.asGraphDef $ sequence $ render <$> toList g
  --print $ graphDef

  let ops = graphDef ^.. node . traverse . name
      --expected = ["Add","Mul","Add","Mul","Const","Add","Const","Const"]
      inps = graphDef ^.. node . traverse . input 
  traverse print $ zip ops inps
  -- putStrLn $ showMessage ops
  return $ dx == V.fromList [6 :: Float] && db == V.fromList [1 :: Float] -- && expected == ops





type Model p a b = forall z. Reifies z W
                => BVar z p
                -> BVar z a
                -> BVar z b


