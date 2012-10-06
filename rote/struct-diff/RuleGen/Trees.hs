{-

This module contains code that is useful for working with trees.
The basic data structure is the Data.Tree structure provided with the
Haskell containers package.  The point of this package is to provide
additional functionality and types built aroudn that data structure that
are useful for the algorithms used in other parts of the program.

matt@galois.com // July 2012

-}
module RuleGen.Trees (
  LabeledTree,
  SizedTree,
  LabeledForest,
  SizedForest,
  toSizedTree,
  fromSizedTree,
  makeSizedForest,
  makeLabeledForest,
  removeSubtrees,
  replaceSubtrees,
  treeToRule
) where

import Data.Tree
import Data.Maybe
import Data.List

-- A tree where the node data is a label
type LabeledTree = Tree String
type LabeledForest = Forest String

-- A tree where the node data is a label and a size representing the number
-- of nodes in the subtree rooted at this node (counting the node itself).
type SizedTree = Tree (String, Int)
type SizedForest = Forest (String, Int)

-- code to turn a labeled tree into a sized tree
toSizedTree :: LabeledTree -> SizedTree
toSizedTree (Node label []) = Node (label, 1) []
toSizedTree (Node label kids) = Node (label, 1+mySize) kids'
  where kids' = map toSizedTree kids
        mySize = foldl1 (+) (map (\(Node (_,count) _) -> count) kids')

-- code to turn a sized tree into a labeled tree
fromSizedTree :: SizedTree -> LabeledTree
fromSizedTree (Node (label,_) kids) = Node label (map fromSizedTree kids)

-- given a subtree size and a tree, return a forest of all subtrees of the
-- tree that have the given size
makeSizedForest :: Int -> SizedTree -> SizedForest
makeSizedForest size n@(Node (_,tSize) kids)
  | tSize > size  = concat $ map (makeSizedForest size) kids
  | tSize == size = [n]
  | otherwise     = []

-- given a label, find all subtrees of a given tree that are rooted with
-- a node with the same label.
makeLabeledForest :: String -> LabeledTree -> LabeledForest
makeLabeledForest lbl t@(Node nlabel kids)
  | lbl == nlabel = [t]
  | otherwise     = concat $ map (makeLabeledForest lbl) kids

-- remove subtrees that are rooted at any node with the provided label.
removeSubtrees :: String -> LabeledTree -> Maybe LabeledTree
removeSubtrees s (Node label kids) 
  | s == label = Nothing
  | otherwise  = Just $ Node label (map fromJust $ filter isJust $ map (removeSubtrees s) kids)

-- replace subtrees that are rooted at a node with the provided label
replaceSubtrees :: String -> LabeledTree -> LabeledTree -> LabeledTree
replaceSubtrees s repl (Node label kids)
  | s == label = repl
  | otherwise  = Node label (map (replaceSubtrees s repl) kids)

-- take a labeled tree, turn it into a textual form that can be used as a
-- part of a stratego rule
treeToRule :: LabeledTree -> String
treeToRule (Node lbl [])   = lbl
treeToRule (Node lbl kids) = lbl++"("++(intercalate "," $ map treeToRule kids)++")"
