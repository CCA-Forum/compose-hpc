{-|
  This module contains code related to filtering data structures.
-}
module RuleGen.Filter (
    despecifyFile,
    generalizeWeave
) where

import RuleGen.Trees
import RuleGen.Weaver
import RuleGen.Pruner
import Data.Tree
import qualified Data.Set as S
import Data.List (foldl')

despecifyFile :: LabeledTree -> LabeledTree
despecifyFile t = replaceSubtrees (LBLString "file_info") replacementNode t
  where replacementNode = Node (LBLString "gen_info()") []

-- =========================
--  EXPERIMENTAL CODE BELOW
--    mjs / 12-2012
-- =========================

type LabelSet = S.Set Label

data Generalization = GSpecific Label [Label]
                    | GClass LabelSet [Label]
  deriving (Show, Eq)

toLabelSet :: [String] -> LabelSet
toLabelSet lbls = S.fromList $ map LBLString lbls

toLabelList :: [String] -> [Label]
toLabelList = map LBLString

arithOps = toLabelSet ["multiply_op", "add_op", "subtract_op", "divide_op"]

generalizeTargets = ["var_ref_exp", "binary_op_annotation"]

genCase = GSpecific (LBLString "multiply_op") (toLabelList generalizeTargets)

generalizeWeave :: WeavePoint -> WeavePoint
generalizeWeave (Mismatch (WLeaf a) (WLeaf b)) = (Mismatch (WLeaf a') (WLeaf b'))
  where (a',b') = generalizer genCase (a,b)
generalizeWeave w = w

generalizer :: Generalization -> (LabeledTree, LabeledTree) -> (LabeledTree, LabeledTree)
generalizer (GClass lsetMatch lsetTargets) _ = error "generalizing equivalence classes unsupported"
generalizer (GSpecific lbl ls) (tIn@(Node rootLbl _), tOut)
  | lbl /= rootLbl = (tIn, tOut)
  | lbl == rootLbl = 
    let
      -- find matching subtrees with correct root labels 
      matches = concat $ map (\l -> makeLabeledForest l tIn) ls

      -- hack
      repls = take (length matches) (map (\i -> Node (LBLString $"TMP_"++(show i)) []) [1..])

      tIn'  = foldl' (\t (templ,repl) -> findAndReplace t templ repl) tIn  (zip matches repls)
      tOut' = foldl' (\t (templ,repl) -> findAndReplace t templ repl) tOut (zip matches repls)
    in (tIn', tOut')