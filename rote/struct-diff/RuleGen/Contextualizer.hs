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

import Data.Maybe (mapMaybe)
import Data.Tree
import Data.Tree.Types
import RuleGen.Util.Configuration
import RuleGen.Util.IDGen
import RuleGen.Weaver
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

--
-- find if a list of weave points contains holes.  if it does,
-- return a list of them.  otherwise, return the empty list
--
holeFinder :: [WeavePoint a] -> [WeavePoint a]
holeFinder []                     = []
holeFinder ((Match _):rest)       = holeFinder rest
holeFinder ((Mismatch _ _):rest)  = holeFinder rest
holeFinder (l@(LeftHole _):rest)  = l:(holeFinder rest)
holeFinder (r@(RightHole _):rest) = r:(holeFinder rest)

ctxtize :: Bool -> (WeavePoint a, Maybe LabeledTree) -> Maybe LabeledTree
ctxtize True  ((LeftHole (WLeaf t)), _)  = Just t
ctxtize False ((LeftHole (WLeaf _)), _)  = Nothing
ctxtize _     ((LeftHole _), _)          = error "Malformed lefthole"
ctxtize True  ((RightHole (WLeaf _)), _) = Nothing
ctxtize False ((RightHole (WLeaf t)), _) = Just t
ctxtize _     ((RightHole _), _)         = error "Malformed righthole"
ctxtize _     (_, lt)                    = lt

checkMatch :: WeavePoint a -> Maybe (WeaveTree a)
checkMatch (Match m) = Just m
checkMatch _         = Nothing

kidVars :: WeavePoint a -> IDGen (WeavePoint a, Maybe LabeledTree)
kidVars k@(LeftHole _)  = return (k,Nothing)
kidVars k@(RightHole _) = return (k,Nothing)
kidVars k               = 
  do l <- strategoVar 
     return (k,Just l)

deeperHoleFinder :: [WeavePoint Bool] -> Bool
deeperHoleFinder []                      = False
deeperHoleFinder ((Match t):rest)        = or [(isHoley t), (deeperHoleFinder rest)]
deeperHoleFinder ((Mismatch lt rt):rest) = or [(isHoley lt), (isHoley rt),
                                              (deeperHoleFinder rest)]
deeperHoleFinder ((LeftHole t):rest)     = or [(isHoley t), (deeperHoleFinder rest)]
deeperHoleFinder ((RightHole t):rest)    = or [(isHoley t), (deeperHoleFinder rest)]

isHoley :: WeaveTree Bool -> Bool
isHoley (WLeaf _)        = False
isHoley (WNode _ flag _) = flag 

kidAnnotator :: WeavePoint a -> WeavePoint Bool
kidAnnotator (Match t)        = Match (holeAnnotator t)
kidAnnotator (Mismatch lt rt) = Mismatch (holeAnnotator lt) (holeAnnotator rt)
kidAnnotator (LeftHole t)     = LeftHole (holeAnnotator t)
kidAnnotator (RightHole t)    = RightHole (holeAnnotator t)

holeAnnotator :: WeaveTree a -> WeaveTree Bool
holeAnnotator (WLeaf t)          = WLeaf t
holeAnnotator (WNode str _ [])   = WNode str False []
holeAnnotator (WNode str _ kids) =
  let holes = not $ null $ holeFinder kids  -- any kids are holes?
      kids' = map kidAnnotator kids
      deeperHoles = deeperHoleFinder kids'
  in WNode str (or [holes, deeperHoles]) kids'

contextualize :: [ContextualizeFilterRule] -> WeaveTree a -> IDGen [(LabeledTree, LabeledTree)]
contextualize rules t = do
  let t' = holeAnnotator t
  results <- mapM (\r -> contextualize_inner r t') rules
  return $ concat results

contextualize_inner :: ContextualizeFilterRule -> WeaveTree Bool -> IDGen [(LabeledTree, LabeledTree)]
-- nothing interesting happens for WLeaf nodes - shouldn't be here
contextualize_inner _ (WLeaf _)                  = error "Can't contextualize a leaf"

-- no holes below here, so nothing interesting will come back.
contextualize_inner _ (WNode _ False _)          = do return []

-- holes be below here.  if this node matches a subtree root to contextualize from,
-- do it!  Otherwise, descend seeking the holes and possible nodes that do
-- match
contextualize_inner cfilt (WNode str True kids)  = do
  let (ContextualizeFilterRule lsetMatch) = cfilt
  if (S.member str lsetMatch) then
    (do kids' <- mapM kidVars kids
        let lhs = Node str (mapMaybe (ctxtize False) kids')
            rhs = Node str (mapMaybe (ctxtize True) kids')
        return [(lhs,rhs)])
    else
      (do rv <- mapM (contextualize_inner cfilt) $ mapMaybe checkMatch kids
          return $ concat rv)
