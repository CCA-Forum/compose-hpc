{-|
  This module contains code that is useful for working with trees.
  The basic data structure is the Data.Tree structure provided with the
  Haskell containers package.  The point of this package is to provide
  additional functionality and types built around that data structure that
  are useful for the algorithms used in other parts of the program.
-}

--
-- matt@galois.com // July 2012
--
module RuleGen.Data.Trees (
  Label(..),
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
  treeToRule,
  dumpTree
) where

import Data.Tree
import Data.Maybe
import Data.List

dumpTree :: Show a => Tree a -> String
dumpTree t = drawTree $ convert t
  where convert (Node n kids) = Node (show n) (map convert kids)
  
{-|
  A tree where the node data is a string label.
-}
data Label = LBLString String
           | LBLList
           | LBLInt Integer
  deriving (Eq, Ord)

instance Show Label where
  show (LBLString s) = s
  show (LBLList)     = "LIST"
  show (LBLInt i)    = show i

type LabeledTree = Tree Label

{-|
  A forest of trees where the node data is a string label.
-}
type LabeledForest = Forest Label

{-|
  A tree where the node data is a label and a size representing the number
  of nodes in the subtree rooted at this node (counting the node itself).
-}
type SizedTree a = Tree (a, Int)

{-|
  A forest of sized trees.
-}
type SizedForest a = Forest (a, Int)

-- | Turn a tree into a sized tree of the same type
toSizedTree :: Tree a       -- ^ Input tree
            -> SizedTree a  -- ^ Output tree corresponding to the input annotated with subtree sizes
toSizedTree (Node label []) = Node (label, 1) []
toSizedTree (Node label kids) = Node (label, 1+mySize) kids'
  where kids' = map toSizedTree kids
        mySize = foldl1 (+) (map (\(Node (_,count) _) -> count) kids')

-- | Turn a sized tree into a regular tree
fromSizedTree :: SizedTree a -- ^ A tree with size annotations
              -> Tree a      -- ^ The input tree with size annotations stripped off
fromSizedTree (Node (label,_) kids) = Node label (map fromSizedTree kids)

{-| 
  Given a subtree size and a tree, return a forest of all subtrees of the
  tree that have the given size.
-}
makeSizedForest :: Int            -- ^ Size of subtrees to extract.
                -> SizedTree a    -- ^ Input tree to chop into sized subtrees.
                -> SizedForest a  -- ^ Forest if subtrees extracted from the input that have the given size.
makeSizedForest size n@(Node (_,tSize) kids)
  | tSize > size  = concat $ map (makeSizedForest size) kids
  | tSize == size = [n]
  | otherwise     = []

{-| 
  Given a label, find all subtrees of a given tree that are rooted with
  a node with the same label.
-}
makeLabeledForest :: Eq a 
                  => a         -- ^ Label to seek.
                  -> Tree a    -- ^ Tree to filter.
                  -> Forest a  -- ^ Forest of subtrees from the input tree with roots matching the label.
makeLabeledForest lbl t@(Node nlabel kids)
  | lbl == nlabel = [t]
  | otherwise     = concat $ map (makeLabeledForest lbl) kids

{-| 
  Remove subtrees that are rooted at any node with the provided label.
-}
removeSubtrees :: Eq a 
               => a               -- ^ Label to seek.
               -> Tree a          -- ^ Tree to filter
               -> Maybe (Tree a)  -- ^ Filtered tree.  Nothing is returned if the label matches the root, otherwise Just t is returned.
removeSubtrees s (Node label kids) 
  | s == label = Nothing
  | otherwise  = Just $ Node label (map fromJust $ filter isJust $ map (removeSubtrees s) kids)

{-|
  Replace subtrees that are rooted at a node with the provided label.
-}
replaceSubtrees :: Eq a 
                => a       -- ^ Label to match.
                -> Tree a  -- ^ Replacement subtree to substitute at nodes where label matches.
                -> Tree a  -- ^ Tree to apply replacement to.
                -> Tree a  -- ^ Tree after replacements have been applied.
replaceSubtrees s repl (Node label kids)
  | s == label = repl
  | otherwise  = Node label (map (replaceSubtrees s repl) kids)

{-| 
  Take a labeled tree, turn it into a textual form that can be used as a
  part of a stratego rule.
-}
treeToRule :: LabeledTree -- ^ Labeled tree to turn into a rule.
           -> String      -- ^ Rule in string form.
treeToRule (Node lbl kids) = 
  let lbl_str = case lbl of
                  LBLInt i    -> show i
                  LBLString s -> s 
                  _           -> ""
      kidstrings = intercalate "," $ map treeToRule kids
  in case kids of
       [] -> case lbl of
               LBLList -> "[]"
               _       -> lbl_str
       _ -> case lbl of
              LBLList -> "["++kidstrings++"]"
              _       -> lbl_str++"("++kidstrings++")"
