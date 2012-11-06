{-|
  This module contains code related to filtering data structures.
-}
module RuleGen.Filter (
    despecifyFile
) where

import RuleGen.Trees
import Data.Tree

despecifyFile :: LabeledTree -> LabeledTree
despecifyFile t = replaceSubtrees (LBLString "file_info") replacementNode t
  where replacementNode = Node (LBLString "gen_info()") []
