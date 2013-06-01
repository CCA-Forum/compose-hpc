{-# LANGUAGE StandaloneDeriving #-}
--
-- antiunification implementation
-- based on implementation used in clonedigger
--
-- matt@galois.com
--
module Data.Tree.AntiUnification (
  antiunify,
  Subs(..),
  Term
) where

import Data.List
import Data.Tree
import Data.Tree.Types
import Control.Monad.State
import qualified Data.Map as Map

-- use state monad for generating unique identifiers
type IDGen = State Int

-- generate sequence of unique ID numbers
genID :: IDGen Int
genID = do i <- get
           put (i+1)
           return i

-- generate variables with a fixed prefix and a unique suffix.  For
-- example, with prefix "x", a sequence of invocations of this
-- function will yield the names "x0", "x1", "x2", and so on.
genName :: String -> IDGen String
genName n = do i <- genID
               return $ n ++ (show i)

type Term = LabeledTree
deriving instance Ord a => Ord (Tree a)

data Subs = Subs (Map.Map Term Term)
  deriving (Show, Eq)

newFreeVar :: IDGen Term
newFreeVar = do n <- genName "VAR"
                return (Node (LBLString n) [])

combineSubs :: (Subs,Subs) -> (Subs,Subs) -> (Subs,Subs)
combineSubs (Subs s0, Subs s1) (Subs t0, Subs t1) =
  (Subs (Map.union s0 t0), Subs (Map.union s1 t1))

antiunifyM :: Term -> Term -> IDGen (Term, Subs, Subs)
antiunifyM n1 n2 | n1 == n2  = return (n1, Subs Map.empty, Subs Map.empty)
antiunifyM n1 n2 | otherwise = do
  let (Node name1 kids1) = n1
      (Node name2 kids2) = n2
  case ((name1 /= name2) || (length kids1 /= length kids2)) of
        True -> do newvar <- newFreeVar 
                   return (newvar, Subs (Map.singleton newvar n1), 
                                   Subs (Map.singleton newvar n2))
        False -> do (kids, subs1, subs2) <- helper kids1 kids2 
                                                   (Subs Map.empty, 
                                                    Subs Map.empty)
                    return (Node name1 kids, subs1, subs2)
  where
    -- ASSUMPTION: two lists passed in to helper are of equal length.
    helper :: [Term] -> [Term] -> (Subs, Subs) -> IDGen ([Term], Subs, Subs)
    helper [] [] (s,t) = return ([],s,t)
    helper xs ys (s,t) = do
      au_out <- mapM (\(x,y) -> antiunifyM x y) (zip xs ys)
      let (nodes,aus,aut) = unzip3 au_out
          (s',t') = foldl' (flip combineSubs) (s,t) (zip aus aut)
      return (nodes,s',t')

antiunify :: Term -> Term -> (Term, Subs, Subs)
antiunify t1 t2 = evalState (antiunifyM t1 t2) 0
