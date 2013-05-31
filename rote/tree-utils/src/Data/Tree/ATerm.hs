{-|
  Helper code to supplement the aterm package on hackage.
  All of our code is in terms of LabeledTrees, which are
  just what we call a forest of string trees using the
  tree and forest type from the standard Data.Tree package.
  The helpers here are useful to turn the aterms into
  these labeled trees.
  Note that this may not be optimal with respect to
  space for huge aterms.  That doesn't matter for the
  purposes of this prototype tool.
-}

module Data.Tree.ATerm (
	readToTree
) where

import ATerm.AbstractSyntax
import ATerm.ReadWrite
import Data.Tree
import Data.Tree.Types

{-|
  Read an aterm from the given filename and return a LabeledTree.
-}
readToTree :: String          -- ^ Filename of aterm file.
           -> IO LabeledTree  -- ^ Aterm converted into a LabeledTree
readToTree fname = do
  t <- readATermFile fname
  return $ atermToTree (getATerm t) t

{-|
  Turn an aterm into a labeled tree.  TODO: handle the ShAList case correctly.
  This will yield bogus labeledtrees as it is, so they will NOT be rendered correctly.
-}
atermToTree :: ShATerm      -- ^ Aterm structure
            -> ATermTable   -- ^ Table of aterm identifiers
            -> LabeledTree  -- ^ Labeled tree created from Aterm structure
atermToTree a t =
  let 
      x = case a of
            (ShAAppl lbl ss _) -> Node (LBLString lbl) (map (\i -> atermToTree (getShATerm i t) t) ss)
            (ShAList ss _) -> Node LBLList (map (\i -> atermToTree (getShATerm i t) t) ss)
            (ShAInt i _) -> Node (LBLInt i) []
  in x
