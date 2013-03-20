{-|
  This module contains code related to filtering data structures.
-}
module RuleGen.Filter (
    preFilter,
    preFilters,
    generalizeWeave
) where

import RuleGen.Data.Trees
import RuleGen.Util.Configuration
import RuleGen.Weaver
import RuleGen.Pruner
import Data.Tree
import Data.List (foldl')
import qualified Data.Set as S

preFilters :: [PreFilterRule] -> LabeledTree -> LabeledTree
preFilters rules tree =
  foldl' (\t f -> preFilter f t) tree rules

preFilter :: PreFilterRule -> LabeledTree -> LabeledTree
preFilter (PreFilterRule roots repl) tree = 
  S.fold (\i t -> replaceSubtrees i replacementNode t) tree roots
  where 
    replacementNode = Node repl []

-- =========================
--  EXPERIMENTAL CODE BELOW
--    mjs / 12-2012
-- =========================

generalizeWeave :: [GeneralizeFilterRule] -> WeavePoint -> WeavePoint
generalizeWeave gens (Mismatch (WLeaf a) (WLeaf b)) = (Mismatch (WLeaf a') (WLeaf b'))
  where (a',b') = foldl' (flip generalizer) (a,b) gens
generalizeWeave _    w = w

--
-- basic generalization process:
--
-- seek subtrees that are rooted at a node labeled with the label carried by the
-- generalization data object.  for each matching rooted subtree, traverse it
-- seeking nodes that have labels appearing in the target set for the
-- generalization object.  for each match, remove the subtree and replace with
-- a metavariable.
--
-- generalization data objects can be either specific subtree roots (GSpecific)
-- or represent an equivalence class of subtree roots that are all to be
-- generalized in the same way.  For example, we may consider all binary
-- arithmetic operators to be equivalent in a generalization sense if we are 
-- working with arithmetic expressions.
--
generalizer :: GeneralizeFilterRule -> (LabeledTree, LabeledTree) -> (LabeledTree, LabeledTree)
generalizer (GeneralizeFilterRule lsetMatch ls) (tIn@(Node rootLbl _), tOut) =
  if (S.member rootLbl lsetMatch) then
    let
      -- find matching subtrees with correct root labels 
      matches = concat $ map (\l -> makeLabeledForest l tIn) ls

      -- hack
      repls = take (length matches) (map (\i -> Node (LBLString $"TMP_"++(show i)) []) ([1..]::[Int]))

      tIn'  = foldl' (\t (templ,repl) -> findAndReplace t templ repl) tIn  (zip matches repls)
      tOut' = foldl' (\t (templ,repl) -> findAndReplace t templ repl) tOut (zip matches repls)
    in (tIn', tOut')
  else
    (tIn, tOut)

