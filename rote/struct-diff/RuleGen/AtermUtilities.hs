{-

Helper code to supplement the aterm package on hackage.
All of our code is in terms of LabeledTrees, which are
just what we call a forest of string trees using the
tree and forest type from the standard Data.Tree package.
The helpers here are useful to turn the aterms into
these labeled trees.

Note that this may not be optimal with respect to
space for huge aterms.  That doesn't matter for the
purposes of this prototype tool.

- matt@galois.com

-}
module RuleGen.AtermUtilities (
	readToTree
) where

import ATerm.ReadWrite
import Data.Tree
import ATerm.AbstractSyntax
import RuleGen.Trees

readToTree :: String -> IO LabeledTree
readToTree fname = do
  t <- readATermFile fname
  return $ atermToTree (getATerm t) t

atermToTree :: ShATerm -> ATermTable -> LabeledTree
atermToTree a t =
  let 
      x = case a of
            (ShAAppl lbl ss _) -> Node lbl (map (\i -> atermToTree (getShATerm i t) t) ss)
            (ShAList ss _) -> Node "LIST" (map (\i -> atermToTree (getShATerm i t) t) ss)
            (ShAInt i _) -> Node (show i) []
  in x