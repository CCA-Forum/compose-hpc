{-|
  code to take two edit trees, and produce a unified tree with them
  woven together.  Each weave point where the trees are attached either
  represents a match, mismatch, or a hole (gap) on either side.

  Author: matt@galois.com
-}
module RuleGen.Weaver (
	WeaveTree(..),
	WeavePoint(..),
	weave,
	forestify,
	toRule
) where

import RuleGen.Yang
import RuleGen.Trees

data WeaveTree = WNode String [WeavePoint]
               | WLeaf LabeledTree
  deriving (Show, Eq)

--
-- a weave point represents a relationship between two subtrees of a larger tree.
-- there are four cases:
--   1. match: the roots of the subtrees match
--   2. mismatch: the roots of the subtrees do not match
--   3. righthole: the best matching by the Yang algorithm determined that the subtree
--      at the given root exists in the start tree and not in the target tree.
--   4. lefthole: similar to righthole, except the root exists in the target tree and
--      not the source.
--
-- [lefthole ==> insertion of new code; righthole ==> deletion of existing code]
--
data WeavePoint = Match WeaveTree
                | Mismatch WeaveTree WeaveTree
                | LeftHole WeaveTree
                | RightHole WeaveTree
  deriving Eq

pp :: WeaveTree -> String
pp (WLeaf t) = treeToRule t
pp t         = error $ "Unexpected pp call for WeaveTree :: "++(show t)

instance (Show WeavePoint) where
	show (Match m)      = "MATCH: "++(show m)++"\n\n"
	show (Mismatch a b) = "MISMATCH: \n   LEFT="++(pp a)++
	                      "\n\n  RIGHT="++(pp b)++"\n\n"
	show (LeftHole m)   = "LH: "++(show m)++"\n\n"
	show (RightHole m)  = "RH: "++(show m)++"\n\n"

toRule :: WeavePoint -> Maybe (String,String)
toRule (Mismatch a b) = Just (pp a, pp b)
toRule _              = Nothing

forestify :: WeaveTree -> [WeavePoint]
forestify (WLeaf _)      = error "forestify encountered WLeaf"
forestify (WNode _ kids) = concat $ map handle kids
  where handle (Match t) = forestify t
        handle x = [x]
	
--	      	
-- given an edit tree, convert ELeaf nodes to WLeaf nodes.  These
-- are used to hang labeled trees off of the tree nodes that represent
-- tree edit operations.  this call makes no sense to make with non-leaf
-- nodes, so it is an error to call them.
--
leafify :: EditTree -> WeaveTree
leafify (ELeaf t) = WLeaf t
leafify x         = error $ "Erroneous leafify call :: "++(show x)

weave :: EditTree -> EditTree -> WeaveTree
weave (ENode albl akids) (ENode _ bkids) = WNode albl (zippy akids bkids)
weave a b                                = error $ "Bad weave :: "++(show a)++
                                                   " // "++(show b)

--
-- this function zips together two lists of editop/edittree pairs.
-- the goal is to yield a single list of "weavepoints", which
-- represent ways that the two trees either match or mismatch.  each
-- case is documented in the code below.
--
zippy :: [(EOp,EditTree)] -> [(EOp, EditTree)] -> [WeavePoint]
--
-- two empty lists, obviously done
--
zippy []              []                = []
--
-- empty list and a delete on the LHS means that there exists a
-- node in the LHS that does not exist in the RHS, so we represent
-- this as a RightHole with the subtree that was deleted from the LHS
-- hanging off of the hole.
--
zippy ((Delete,t):xs) []                = (RightHole $ leafify t)   :(zippy xs [])
--
-- symmetric with previous case : this represents missing nodes on the
-- the LHS, and a correspond left hole.
--
zippy [] ((Delete,t):ys)                = (LeftHole $ leafify t)    :(zippy [] ys)
--
-- if both sides say keep, we have a match.  recurse down into the matching
-- trees
--
zippy ((Keep,xt):xs) ((Keep,yt):ys)     = (Match (weave xt yt))     :(zippy xs ys)
--
-- both sides disagree, so we delete.  this represents a mismatch, which
-- we hang the two mismatching trees off of.
--
zippy ((Delete,xt):xs) ((Delete,yt):ys) = (Mismatch (leafify xt) (leafify yt)):
                                          (zippy xs ys)
--
-- deletion on one side, but no delete on the other side (previous case didn't
-- match), so we assume a right hole
--
zippy ((Delete,xt):xs) ys               = (RightHole $ leafify xt)  :(zippy xs ys)
--
-- symmetric with previous case.
--
zippy xs ((Delete,yt):ys)               = (LeftHole $ leafify yt)   :(zippy xs ys)
--
-- by construction of the dynamic programming table in the Yang algorithm,
-- when one tree runs out of nodes, the rest are deleted.  it is impossible
-- to produce a "keep" action in this case, unless there is a bug in the
-- traceback logic in the yang code.  this error is unrecoverable, since it
-- means we have a fatal flaw in the tree matching algorithm.
--
zippy [] ((Keep,_):_)                   = error "zippy hit [] ((Keep,_):_) case"
--
-- symmetric pathological case
--
zippy ((Keep,_):_) []                   = error "zippy hit ((Keep,_):_) [] case"
