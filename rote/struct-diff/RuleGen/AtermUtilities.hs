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