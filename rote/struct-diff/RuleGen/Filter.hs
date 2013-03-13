{-|
  This module contains code related to filtering data structures.
-}
module RuleGen.Filter (
    despecifyFile,
    generalizeWeave,
    readGeneralizationConfig,
    Generalization (..)
) where

import Data.ConfigFile
import RuleGen.Data.Trees
import RuleGen.Weaver
import RuleGen.Pruner
import Data.Tree
import qualified Data.Set as S
import Data.List (foldl')
import RuleGen.Util.Misc

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

generalizeWeave :: [Generalization] -> WeavePoint -> WeavePoint
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
generalizer :: Generalization -> (LabeledTree, LabeledTree) -> (LabeledTree, LabeledTree)
generalizer (GClass lsetMatch ls) (tIn@(Node rootLbl _), tOut) =
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

generalizer (GSpecific lbl ls) (tIn@(Node rootLbl _), tOut)
  | lbl /= rootLbl = (tIn, tOut)
  | lbl == rootLbl = 
    let
      -- find matching subtrees with correct root labels 
      matches = concat $ map (\l -> makeLabeledForest l tIn) ls

      -- hack
      repls = take (length matches) (map (\i -> Node (LBLString $"TMP_"++(show i)) []) ([1..]::[Int]))

      tIn'  = foldl' (\t (templ,repl) -> findAndReplace t templ repl) tIn  (zip matches repls)
      tOut' = foldl' (\t (templ,repl) -> findAndReplace t templ repl) tOut (zip matches repls)
    in (tIn', tOut')

generalizer _ _ = error "Bad generalizer call"

--
-- read a configuration file of generalizations.  the format is as follows:
--
-- [sectionname]
-- root : node_name, node_name, ..., node_name
-- target :: node_name, node_name, ..., node_name
--
-- both the root and target entries can be one or more items.
--
-- TODO: better error handling than the forceEither method, which is
--       not really graceful in dealing with badness
--
readGeneralizationConfig :: String -> IO [Generalization]
readGeneralizationConfig fname = do
    val <- readfile emptyCP fname
    let cp = forceEither val
        sects = sections cp
        csvs s = map (filter (/=' ')) $ wordsWhen (==',') s
        handleSection s = let root = csvs $ forceEither $ get cp s "root"
                              targets = csvs $ forceEither $ get cp s "target"
                          in case root of 
                               []     -> error "Fatal error in generalization config."
                               (x:[]) -> GSpecific (LBLString x) (toLabelList targets)
                               _      -> GClass (toLabelSet root) (toLabelList targets)
    return $ map handleSection sects
