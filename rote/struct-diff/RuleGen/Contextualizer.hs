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
import Data.Maybe
import RuleGen.Weaver

--
-- TODO: replace with proper variable and symbol generator
--
strategoVar :: LabeledTree
strategoVar = Node (LBLString "STRATEGOVAR") []

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

ctxtize :: Bool -> WeavePoint -> Maybe LabeledTree
ctxtize True  (LeftHole (WLeaf t))  = Just t
ctxtize False (LeftHole (WLeaf _))  = Nothing
ctxtize _     (LeftHole _)          = error "Malformed lefthole"
ctxtize True  (RightHole (WLeaf _)) = Nothing
ctxtize False (RightHole (WLeaf t)) = Just t
ctxtize _     (RightHole _)         = error "Malformed righthole"
ctxtize _     _                     = Just strategoVar

checkMatch :: WeavePoint -> Maybe WeaveTree
checkMatch (Match m) = Just m
checkMatch _         = Nothing

unmaybeList :: [Maybe a] -> [a]
unmaybeList l = map fromJust $ filter isJust l

contextualize :: WeaveTree -> [(LabeledTree, LabeledTree)]
-- nothing interesting happens for WLeaf nodes - shouldn't be here
contextualize (WLeaf _)        = error "Can't contextualize a leaf"
-- the action is in the WNodes...
contextualize (WNode str kids) =
  let
    -- if any of the kids of this node are a hole, then we build
    -- context from this node as parent.
    holes = holeFinder kids

  in case holes of
     	 -- no kids are holes, so descend into subtrees that match
    	 [] ->  concat $ map contextualize $ unmaybeList (map checkMatch kids)

       -- we have one or more kid-holes.  what to do?
       --  1. all non-hole kids, they become stratego variables
       --  2. all hole kids are emitted as their raw trees
    	 _  -> [(Node str (unmaybeList $ map (ctxtize False) kids),
               Node str (unmaybeList $ map (ctxtize True) kids))]

    	