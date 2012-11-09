{-|
  This module contains code related to pruning trees to make it easier to
  represent rewrite rules.  For example, one type of operation we would
  want to perform on a tree is to identify all variable reference
  expressions, and abstract them out to be replaced with Stratego variables
  allowing them to generalize.
-}
module RuleGen.Pruner (
  findAndReplace,
  getLabeledSubtrees,
  variableReplacer
) where

import RuleGen.Yang
import RuleGen.Trees
import Data.Tree

-- given a subtree rooted at a var_ref_exp, get the name
-- of the variable being referenced.  the danger referred
-- to by the function name is the assumption we make and
-- hardcode into the pattern match about where the name
-- resides in the tree.  DANGER!
dangerousVariableNameGetter :: LabeledTree -> String
dangerousVariableNameGetter t =
	let (Node _ ((Node _ (_:(Node (LBLString s) _):_)):_)) = t
	in s

makeReplacements :: LabeledForest -> [(LabeledTree,LabeledTree)]
makeReplacements [] = []
makeReplacements (t:ts) = 
  let repl = Node (LBLString ("STRATEGO_"++(dangerousVariableNameGetter t))) []
  in (t,repl):(makeReplacements ts)

{-|
  This function takes the tree representing the source and the tree
  representing the target, and performs the following steps: all variable
  references in the source are identified, and replacements are generated
  for them.  Once this has taken place, each replacement is made in both
  the source and target trees.  The replacements are common to both
  trees such that corresponding variables match between the trees, and
  therefore on both sides of the rewrite rule.
-}
variableReplacer :: LabeledTree -- ^ Source tree
                 -> LabeledTree -- ^ Target tree
                 -> (LabeledTree, LabeledTree) -- ^ Modified source and target
variableReplacer src targ =
  let 
    s_vrefs = getLabeledSubtrees (LBLString "var_ref_exp") src
    repls = makeReplacements s_vrefs

    replacer t [] = t
    replacer t ((v,r):rest) =
      let t' = findAndReplace t v r
      in replacer t' rest

    src' = replacer src repls
    targ' = replacer targ repls
  in (src', targ')

{-|
  Given a string and a labeled tree, produce a forest of labeled trees
  representing all subtrees of the input tree that are labeled with the
  given string at their root.  Note that in a top-down traversal, once a
  match is found, the entire subtree is put into the forest and no further
  traversal is performed into the subtree.  Therefore each tree in the
  forest can be safely considered to represent distinct, non-overlapping
  subtrees of the original tree.
-}
getLabeledSubtrees :: Label        -- ^ Label to search for
                   -> LabeledTree   -- ^ Tree to search
                   -> LabeledForest -- ^ Set of subtrees of the large tree with roots matching the provided label
getLabeledSubtrees label wholetree =
  let
     -- first, find all of the label-rooted trees
     f = makeLabeledForest label wholetree

     -- now, filter out non-unique trees from the forest
     filterer _ _ [] = []
     filterer t tsize (c:cs) = 
     	let score = treedist t c (==)
        in if (score == tsize) then filterer t tsize cs
                               else c:(filterer t tsize cs)

     filterforest [] = []
     filterforest (t:ts) =
     	let Node (_,tsize) _ = toSizedTree t
     	in t:filterforest (filterer t (fromIntegral tsize) ts)

     funique = filterforest f
  in
    funique

{-|
  This function takes a tree to transform, a template to find all instances
  of within the tree, and a replacement subtree to use to replace all
  occurances of the template in the original tree.
-}
findAndReplace :: LabeledTree  -- ^ Tree to transform
               -> LabeledTree  -- ^ Template to search for
               -> LabeledTree  -- ^ Replacement subtree for all matching instances of template
               -> LabeledTree  -- ^ Transformed tree
findAndReplace bigtree templatetree replacementtree =
  let
     -- turn our trees into sized equivalents for internal use
     sbigtree = toSizedTree bigtree
     stemplate = toSizedTree templatetree

     -- how big is the template?
     Node (_,templatesize) _ = stemplate

     -- now, do a traversal looking for candidate subtrees (matching size)
     --
     -- IMPORTANT NOTE: the replacement tree may not have the same size as the
     --                 template being matched.  as a result, we allow the sized
     --                 trees to "go bad" as subtrees are replaced.  no repair is
     --                 made since we don't expose the size information outside
     --                 the function once the template matching has occurred.
     --
     walker n@(Node (lbl,siz) kids)
       | siz == templatesize = let yscore = treedist (fromSizedTree n) templatetree (==)
                               in if (yscore == (fromIntegral templatesize)) 
                               	    then [toSizedTree $ replacementtree]
                               	    else [n]
       | siz < templatesize  = [n]
       | otherwise           = [Node (lbl,siz) (concat $ map walker kids)]

  in fromSizedTree $ head $ walker sbigtree

