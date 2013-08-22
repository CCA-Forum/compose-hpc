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

import Data.Maybe (catMaybes, mapMaybe)
import Data.Tree
import Data.Tree.Types
import Data.Tree.Weaver
import RuleGen.Util.Configuration
import RuleGen.Util.IDGen
import qualified Data.Set as S

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

-- annotate a weave tree with a bool flag that indicates
-- whether or not a subtree of each node contains a hole.
containsHoles :: WeaveTree a -> WeaveTree Bool
containsHoles (WLeaf t) = WLeaf t
containsHoles (WNode lbl _ kids) = WNode lbl (or flags) kids'
  where kfilt (Match m) = let w@(WNode _ flag _) = containsHoles m
                          in (Match w, flag)
        kfilt (Mismatch m1 m2) = (Mismatch (containsHoles m1) (containsHoles m2), False)
        kfilt (LeftHole lh) = (LeftHole (containsHoles lh), True)
        kfilt (RightHole rh) = (RightHole (containsHoles rh), True)
        (kids',flags) = unzip $ map kfilt kids

-- wrapper that annotates the tree with booleans indicating the presence of a hole in
-- subtrees, and then calls the main function that does the traversal called
-- contextualize_booltree
contextualize :: [ContextualizeFilterRule] -> WeaveTree a -> IDGen [(LabeledTree, LabeledTree)]
contextualize ctxt_filt t = do
  let bt = containsHoles t
      lsetmatch = foldl1 S.union $ map (\(ContextualizeFilterRule r) -> r) ctxt_filt
  contextualize_booltree (ContextualizeFilterRule lsetmatch) bt

-- main function that does the traversal.  spins down until it hits a node that has a hole
-- under it, AND has a matching label.  if this occurs, it hands off to the handler for the
-- subtree.
contextualize_booltree :: ContextualizeFilterRule -> WeaveTree Bool 
               -> IDGen [(LabeledTree, LabeledTree)]
contextualize_booltree ctxt_filt@(ContextualizeFilterRule filt_set) w@(WNode lbl True kids) = do
  if (S.member lbl filt_set) then
    contextualize_inner w
    else
      (do rv <- mapM (contextualize_booltree ctxt_filt) $ mapMaybe checkMatch kids
          return $ concat rv )
contextualize_booltree _ (WNode _ False _) = do return []
contextualize_booltree _ (WLeaf _) = error "BAD"

checkMatch :: WeavePoint a -> Maybe (WeaveTree a)
checkMatch (Match m) = Just m
checkMatch _         = Nothing

contextualize_inner :: WeaveTree Bool 
                    -> IDGen [(LabeledTree, LabeledTree)]
-- WLeafs can be assumed by construction to only hang off specific kinds of
-- weave points, and are handled elsewhere.  This function should only get
-- invoked for nodes that contain WNode entries.
contextualize_inner (WLeaf _) = 
  error "Contextualize_inner incorrectly reached a WLeaf."

-- no holes underneath, and we are inside a subtree that is under a context
-- node, so subtrees without holes turn into metavariables.
contextualize_inner (WNode _ False _) = do  
  l <- strategoVar  -- TODO: is this right?  leave subtrees instead 
                    --       to make most specific instead of most general?
  return [(l,l)]

-- otherwise, handle kids and keep this label
contextualize_inner (WNode lbl True kids) = do
  kids' <- mapM (ctxtWP) kids
  let (kl,kr) = unzip $ concat kids'
  return [(Node lbl $ catMaybes kl, Node lbl $ catMaybes kr)]

ctxtWP :: WeavePoint Bool -> IDGen [(Maybe LabeledTree, Maybe LabeledTree)]
ctxtWP (LeftHole (WLeaf lh))            = return [(Nothing, Just lh)]
ctxtWP (RightHole (WLeaf rh))           = return [(Just rh, Nothing)]
ctxtWP (Mismatch (WLeaf m1) (WLeaf m2)) = return [(Just m1, Just m2)]
ctxtWP (Match n)                        = 
  do xs <- contextualize_inner n
     let mxs = map (\(a,b) -> (Just a, Just b)) xs
     return mxs
ctxtWP _                                = error "Malformed weavepoint encountered"