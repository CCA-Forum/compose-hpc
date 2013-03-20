{-|
  This module contains code related to filtering data structures.
-}
module RuleGen.Filter (
    preFilter,
    preFilters
) where

import RuleGen.Data.Trees
import RuleGen.Util.Configuration
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

