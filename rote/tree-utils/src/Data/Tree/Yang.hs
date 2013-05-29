--
-- implementation of tree matching from yang '91 paper
--
-- matt@galois.com
--
module Data.Tree.Yang (
  treediff,
  treedist,
  EditTree(..),
  EOp(..),
  LabelComparator,
  etreeToLTree,
  replaceEditTreeNode
) where

import Data.Array
import Data.Maybe
import Data.List
import Data.Tree
import Data.Tree.Types
import Prelude hiding (Left)

-- edit node is used to represent the original tree node with its  
-- children enumerated with a corresponding keep/delete operation
-- determining if they stay or go in the matching
data EditTree = ENode Label [(EOp,EditTree)]
              | ELeaf LabeledTree  -- can get subtree out of the original Tree
              | ENil               -- used for the zero row/column of matrix
  deriving (Show, Eq)

etreeToLTree :: EditTree -> LabeledTree
etreeToLTree (ELeaf l)        = l
etreeToLTree (ENode lbl kids) = Node lbl $ map (\(_,e) -> etreeToLTree e) kids
etreeToLTree ENil             = error "Fatal: ENil encountered in the wild"

replaceEditTreeNode :: Label -> EditTree -> LabeledTree -> EditTree -> EditTree
replaceEditTreeNode lbl replET replLT t =
  case t of
    ENil                     -> ENil
    ELeaf e                  -> ELeaf (replaceSubtrees lbl replLT e)
    ENode s kids | s == lbl  -> replET
                 | otherwise -> ENode s (map (\(o,k) -> (o,replaceEditTreeNode lbl replET replLT k)) kids)

cleaner :: EditTree -> EditTree
cleaner (ELeaf t)        = (ELeaf t)
cleaner ENil             = ENil
cleaner (ENode lbl kids) = ENode lbl (map sanitize kids)

sanitize :: (EOp,EditTree) -> (EOp,EditTree)
sanitize (Keep,t)   = (Keep,   t)
sanitize (Delete,t) = (Delete, ELeaf (reconstruct t))

reconstruct :: EditTree -> LabeledTree
reconstruct (ELeaf t)      = t
reconstruct (ENil)         = undefined
reconstruct (ENode l kids) = Node l (map (\(_,kid) -> reconstruct kid) kids)

-- edit operation
data EOp = Keep | Delete
  deriving (Show, Eq)

-- traceback help
data Direction = Left | Up | Diag
  deriving (Show, Eq)

-- type for a label comparison function
type LabelComparator = Label -> Label -> Bool

treediff :: LabeledTree -> LabeledTree -> LabelComparator -> (EditTree,EditTree)
treediff t1 t2 labelcompare = (cleaner y1, cleaner y2)
  where (_,(y1,y2)) = yang t1 t2 labelcompare

treedist :: LabeledTree -> LabeledTree -> LabelComparator -> Integer
treedist t1 t2 labelcompare = score
  where (score,_) = yang t1 t2 labelcompare

-- given two trees, return a similarity score and the two trees converted
-- into edit trees
yang :: LabeledTree -> LabeledTree -> LabelComparator -> (Integer, (EditTree, EditTree))
yang ta tb labelcompare = (score, (reta, retb))
  where
    -- unpack data from a node
    (Node alabel akids) = ta
    (Node blabel bkids) = tb

    -- helper to turn traceback directions into delete/keep operations
    dirToOp Up   = (Just Delete, Nothing)
    dirToOp Left = (Nothing, Just Delete)
    dirToOp Diag = (Just Keep, Just Keep)

    -- compute traceback from lower right corner (lena,lenb)
    -- traceback is list of [(x,y),Dir,(l,r)] where (x,y) is the location
    -- in the table, Dir is the direction, and (l,r) are the computed
    -- yang results for the subtrees (or, ELeaf if the node is deleted and
    -- the subtree not traversed) 
    eseq = reverse $ traceback lena lenb

    eseq' = map (\(_,d,(l,r)) -> let (opl,opr) = dirToOp d
                                 in ((opl,l),(opr,r))) 
                eseq
    (tba,tbb) = unzip eseq'
    aekids = map (\(x,y) -> (fromJust x,y)) $ filter (\(x,_) -> isJust x) tba
    bekids = map (\(x,y) -> (fromJust x,y)) $ filter (\(x,_) -> isJust x) tbb

    -- get number of first level subtrees
    lena = length akids
    lenb = length bkids

    -- score for node - use provided comparator
    score = if (labelcompare alabel blabel) then 1 + (lena @@ lenb)
                                            else 0 

    (reta, retb) = if (score==0) then (ELeaf ta, ELeaf tb)
                                 else (ENode alabel aekids, 
                                       ENode blabel bekids)

    -- arrays
    ak = listArray (1,lena) akids
    bk = listArray (1,lenb) bkids

    -- define the lazy table.  
    ytable = listArray ((0,0),(lena,lenb))
                        [yscore x y | x <- [0 .. lena], y <- [0 .. lenb]] 
                        :: Array (Int,Int) (Integer,Direction,(EditTree,EditTree))

    --scores = [((i,j),i@@j) | i <- [0..lena], j <- [0..lenb]]

    -- create an infix operator to make the accesses to the table
    -- syntactically nicer, instead of peppering the code with the
    -- ugly "ytable ! (i,j)" type code.
    infix 5 @@
    (@@) i j = let (a,_,_) = ytable ! (i, j) in a
    infix 5 ##
    (##) i j = let (_,b,_) = ytable ! (i, j) in b
    infix 5 <>
    (<>) i j = let (_,_,c) = ytable ! (i, j) in c

    maxer (a,x,m) (b,y,n) | a > b     = (a,x,m)
                          | otherwise = (b,y,n) 

    maxer3 a b c = maxer (maxer a b) c

    yscore 0 0 = (0, Diag, (ENil         , ENil))
    yscore 0 j = (0, Left, (ENil         , ELeaf (bk!j)))
    yscore i 0 = (0, Up  , (ELeaf (ak!i) , ENil))
    yscore i j = maxer3 (((i-1 @@ j-1) + ijscore), Diag, (ijl, ijr))
                        ((i-1 @@ j),               Up,   (ijl, ijr))
                        ((i @@ j-1),               Left, (ijl, ijr))
      where (ijscore,(ijl, ijr)) = yang (ak!i) (bk!j) labelcompare

    traceback 0 0 = []
    traceback x y =
       case move of
           Up   -> ((x,y),Up,(l,r))   : traceback (x-1) y
           Left -> ((x,y),Left,(l,r)) : traceback x (y-1)
           Diag -> ((x,y),Diag,(l,r)) : traceback (x-1) (y-1)
      where
        move = (x ## y)
        (l,r) = (x <> y)
