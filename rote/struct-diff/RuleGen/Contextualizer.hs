{-|
  This module contains code for taking parts of the tree that
  we wish to turn into rewrite rules that require context.
  Specificially, when we have holes (left or right), we need
  to consider the parent of the hole to provide the context on
  both sides of the rewrite rule defining where the appropriate
  code insertion/deletion occurs.
-}
module RuleGen.Contextualizer (
  contextualize
) where

import RuleGen.Trees
import Data.Tree
import Data.Maybe (mapMaybe)
import RuleGen.Weaver
import RuleGen.IDGen

--
-- generate unique labeled tree elements.  each element represents
-- a leaf node in the tree -- a string-labeled node with an empty
-- set of children.  the name of the node is "RG_" followed by a
-- unique integer drawn from the pool managed by the IDGen monad.
--  (RG = RuleGen)
--
strategoVar :: IDGen LabeledTree
strategoVar = do
  ident <- genName "RG_"
  return $ Node (LBLString ident) []

--
-- find if a list of weave points contains holes.  if it does,
-- return a list of them.  otherwise, return the empty list
--
holeFinder :: [WeavePoint] -> [WeavePoint]
holeFinder []                     = []
holeFinder ((Match _):rest)       = holeFinder rest
holeFinder ((Mismatch _ _):rest)  = holeFinder rest
holeFinder (l@(LeftHole _):rest)  = l:(holeFinder rest)
holeFinder (r@(RightHole _):rest) = r:(holeFinder rest)

ctxtize :: Bool -> (WeavePoint, Maybe LabeledTree) -> Maybe LabeledTree
ctxtize True  ((LeftHole (WLeaf t)), _)  = Just t
ctxtize False ((LeftHole (WLeaf _)), _)  = Nothing
ctxtize _     ((LeftHole _), _)          = error "Malformed lefthole"
ctxtize True  ((RightHole (WLeaf _)), _) = Nothing
ctxtize False ((RightHole (WLeaf t)), _) = Just t
ctxtize _     ((RightHole _), _)         = error "Malformed righthole"
ctxtize _     (_, lt)                    = lt

checkMatch :: WeavePoint -> Maybe WeaveTree
checkMatch (Match m) = Just m
checkMatch _         = Nothing

kidVars :: WeavePoint -> IDGen (WeavePoint, Maybe LabeledTree)
kidVars k@(LeftHole _)  = return (k,Nothing)
kidVars k@(RightHole _) = return (k,Nothing)
kidVars k               = 
  do l <- strategoVar 
     return (k,Just l)

contextualize :: WeaveTree -> IDGen [(LabeledTree, LabeledTree)]
-- nothing interesting happens for WLeaf nodes - shouldn't be here
contextualize (WLeaf _)        = error "Can't contextualize a leaf"

-- the action is in the WNodes...
contextualize (WNode str kids) = do
  -- if any of the kids of this node are a hole, then we build
  -- context from this node as parent.
  let holes = holeFinder kids

  case holes of
    -- no kids are holes, so descend into subtrees that match
    [] -> do rv <- mapM contextualize $ mapMaybe checkMatch kids
             return $ concat rv

    -- we have one or more kid-holes.  what to do?
    --  1. all non-hole kids, they become stratego variables
    --  2. all hole kids are emitted as their raw trees
    _  -> do kids' <- mapM kidVars kids
             let lhs = Node str (mapMaybe (ctxtize False) kids')
                 rhs = Node str (mapMaybe (ctxtize True) kids')
             return [(lhs,rhs)]