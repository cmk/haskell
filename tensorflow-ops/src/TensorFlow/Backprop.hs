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

import Control.Monad (replicateM_, void)   

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
import qualified TensorFlow.Build as B
import GHC.IO     ( IO(..) )

import TensorFlow.Types (unScalar)
import TensorFlow.Output (Device(..))
import TensorFlow.Tensor
import TensorFlow.Session
    ( run
    , runSession
    , run_
    , Session
    )
import qualified TensorFlow.Session as S
import Data.Int (Int8, Int16, Int32, Int64)

import TensorFlow.Gradient (gradients)
import TensorFlow.ControlFlow
import Data.ProtoLens.TextFormat (showMessage)
--import Proto.Tensorflow.Core.Framework.Graph (node)
--import Proto.Tensorflow.Core.Framework.NodeDef (op, input, name)

import Control.Monad.Primitive
import Control.Monad.ST
import Data.STRef

import Proto.Tensorflow.Core.Framework.Graph_Fields (node)
import Proto.Tensorflow.Core.Framework.NodeDef_Fields (op, input, name)

import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Variable as TF

import Control.Monad.Reader   ( ReaderT, lift )
import Unsafe.Coerce
{-
https://mail.haskell.org/pipermail/haskell/2007-May/019540.html

TODO reimplement functions in https://tensorflow.github.io/haskell/haddock/tensorflow-ops-0.2.0.0/TensorFlow-Variable.html
to use PrimMonad instead of MonadBuild


instance PrimMonad m => PrimMonad (ReaderT r m) where
  type PrimState (ReaderT r m) = PrimState m
  primitive = lift . primitive

-- need one of these
instance PrimMonad m => MonadBuild m
instance PrimMonad (SessionT (ST s))

instance PrimMonad m => PrimMonad (ReaderT r m) where
  type PrimState (ReaderT r m) = PrimState m
  primitive = lift . primitive

newtype SessionT m a = Session (ReaderT SessionState (BuildT m) a)
type SST s a = S.SessionT (ST s) a

instance PrimMonad (S.SessionT (ST s)) where
  type PrimState (S.SessionT (ST s)) = s
  primitive = lift . primitive . unsafeCoerce 

> :t runIdentity . runBuildT
runIdentity . runBuildT :: BuildT Identity a -> (a, GraphState)

instance MonadBuild (ST s) where
  build b = do
    r <- newSTRef initGraphState
    rin <- readSTRef r
    let (a,rout) = runIdentity . runStateT (unsafeCoerce b) $ rin
    writeSTRef r rout
    return a 

-}

--newtype Mut a = Mut { unMut :: forall r. ReaderT (STRef r [TF.ControlNode]) (ST r) a } 
type Mutable s a = ReaderT (STRef s [TF.ControlNode]) (ST s) a 

read' :: (PrimMonad m) => Int -> m Int
read' = return

foo :: Mutable s Int
foo = read' 9

--newtype MT s a = MTensor { unMTensor :: forall v. Tensor v a } 
newtype MT s a = MTensor { unMTensor :: TF.Variable a } 
type T a = Tensor Build a

type Model p a b = forall z. Reifies z W => BVar z p -> BVar z a -> BVar z b

{-


gradientDescent :: a -> [TF.Variable a] -> [TF.Tensor TF.Value a] -> m TF.ControlNode
-- | Perform one step of the gradient descent algorithm.
gradientDescent :: TVal -> [Variable TVal] -> [Tensor Value TVal] -> Build ControlNode
gradientDescent learningRate params grads = TF.withNameScope "gradientDescent" $ do

  let applyGrad param grad = TF.assignAdd param (TF.scalar (-0.1) `TF.mul` grad) -- this in ST
  TF.group =<< zipWithM applyGrad params grads
-}


freeze :: TF.TensorType a => MT s a -> ST s (T a)
freeze = return . TF.readValue . unMTensor

(+=) :: TF.TensorType a => MT s a -> T a -> ST s ()
(+=) (MTensor v) t = void $ TF.assignAdd v t

gradBP2'' :: Model (T Float) (T Float) (T Float) -> MT s Float -> MT s Float -> ST s (T Float, T Float)
gradBP2'' f a b = do
  a' <- freeze a
  b' <- freeze b
  return $ gradBP2 f a' b'


gradBP2' 
  :: TF.MonadBuild m
  => (forall s. Reifies s W => BVar s (T Float) -> BVar s (T Float) -> BVar s (T Float))
  -> [TF.Variable Float] -> m [Tensor Value Float]
gradBP2' f [a,b] = sequence $ fmap render [c,d]  where (c, d) = gradBP2 f (TF.readValue a) (TF.readValue b)

-- TODO use STRef? http://hackage.haskell.org/package/base-4.3.1.0/docs/Data-STRef.html
minimize'
  :: TF.MonadBuild m
  => (forall s. Reifies s W => BVar s (T Float) -> BVar s (T Float) -> BVar s (T Float))
  -> [TF.Variable Float] -> m TF.ControlNode -- m ()
minimize' f params = gradBP2' f params >>= TF.gradientDescent 0.01 params

minimize'' :: Model (T Float) (T Float) (T Float) -> MT s Float -> MT s Float -> ST s ()
minimize'' f a b = gradBP2'' f a b >>= gradientDesc 0.01 a b

gradientDesc :: Float -> MT s Float -> MT s Float -> (T Float, T Float) -> ST s ()
gradientDesc = undefined

--minimize' lossyeah :: TF.MonadBuild m => [TF.Variable Float] -> m TF.ControlNode

ones = [1, 1, 1, 1] :: [Float]
matx = TF.constant [2, 2] ones
f x y = (constVar matx) `subD` (x `matMulD` y)

lossyeah :: Model (T Float) (T Float) (T Float) 
lossyeah x y = myMeanD $ squareD $ f x y

fuckyeah :: [TF.Variable Float] -> Session [Tensor Value Float]
fuckyeah = gradBP2' lossyeah

randomParam :: TF.Shape -> TF.Session (TF.Tensor TF.Value Float)
randomParam (TF.Shape shape) = TF.truncatedNormal (TF.vector shape)



fitMatrix = TF.runSession $ do
  u <- TF.initializedVariable =<< randomParam [2, 1]
  v <- TF.initializedVariable =<< randomParam [1, 2]
  let ones = [1, 1, 1, 1] :: [Float]
      matx = TF.constant [2, 2] ones

      --f x y = (constVar matx) `subD` (x `matMulD` y)
      --gradBP2' loss

      diff = matx `C.sub` (TF.readValue u `C.matMul` TF.readValue v)
      loss = TF.reduceMean $ C.square diff

      --f :: [Tensor Value TVal] -> Build ControlNode
      --f params = TF.gradientDescent 0.01 params
      --minimize :: Tensor Build TVal	-> [Variable TVal]	-> Build ControlNode
      --minimize loss params = gradients loss params >>= TF.gradientDescent 0.01 params
      minimize params = fuckyeah params >>= TF.gradientDescent 0.01 params

      
  trainStep <- minimize [u, v]
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



type MT s d = MTensor s d TVal
newtype MTensor s d e = MTensor { unMTensor :: MVector s e } -- vector 
newtype MTensor s d e = MTensor { unMTensor :: Variable e } -- tensorflow



-- TODO : ok to go : Tensor Value TVal -> Tensor Build TVal -> Tensor Value TVal ??
-- renderValue . expr == id ??
gradients2 :: PrimMonad m => Model -> [MT s d] -> ST s [T d]


minimize :: Tensor Build TVal	-> [Variable TVal]	-> Build ControlNode
minimize loss params = TF.gradients loss params >>= f
  where  f :: [Tensor Value TVal] -> Build ControlNode
         f = gradientDescent 0.01 params

newtype MT s a = MTensor { unMTensor :: TF.Variable a } 
type T a = Tensor Build a

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


eval :: TF.Fetchable t a => t -> IO a
eval = TF.runSession . TF.run


minusP :: Tensor Build Float -> (Tensor Build Float, Tensor Build Float)
       -> (Tensor Build Float, Tensor Build Float) 
       -> (Tensor Build Float, Tensor Build Float)
minusP eps (a,b) (c,d) = (a-eps*c, b-eps*d)


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







