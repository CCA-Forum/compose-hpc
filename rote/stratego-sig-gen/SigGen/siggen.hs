--
-- quick stratego signature generator
--
-- matt@galois.com
--
import ATerm.ReadWrite
import Data.Tree
import ATerm.AbstractSyntax
import Data.Maybe
import Data.List
import System.Environment

data Constructor = Constructor String [String]
  deriving Show

-- need this so nub will work
instance (Eq Constructor) where
    (==) (Constructor a _) (Constructor b _) = a==b

-- filter a list of subtrees to either be E or [E]
kidMap :: [LabeledTree] -> [String]
kidMap []         = []
kidMap (cur:rest) = 
    let root = rootLabel cur
        rtrm = case root of
                 LBLList -> "[E]"
                 _       -> "E"
    in rtrm:(kidMap rest)

processOneNode :: LabeledTree -> Maybe Constructor
processOneNode n =
    let root = rootLabel n
        kids = subForest n
        km = kidMap kids
    in
        case root of
            LBLString s -> Just $ Constructor s km
            LBLInt i    -> Just $ Constructor (show i) km
            LBLList     -> Nothing

processTree :: LabeledTree -> [Maybe Constructor]
processTree t =
    let kids = subForest t
    in (processOneNode t):(concat $ map processTree kids)

data Label = LBLString String
           | LBLList
           | LBLInt Integer
  deriving (Eq, Ord)

instance Show Label where
  show (LBLString s) = s
  show (LBLList)     = "LIST"
  show (LBLInt i)    = show i

type LabeledTree = Tree Label

readToTree :: String          -- ^ Filename of aterm file.
           -> IO LabeledTree  -- ^ Aterm converted into a LabeledTree
readToTree fname = do
  t <- readATermFile fname
  return $ atermToTree (getATerm t) t

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

prettyRules :: [Constructor] -> [String]
prettyRules [] = []
prettyRules ((Constructor _ []):rest) = prettyRules rest
prettyRules ((Constructor lbl kids):rest) =
    (lbl++" : "++(intercalate " * " kids)++" -> E"):(prettyRules rest)

main :: IO ()
main = do
    args <- getArgs
    let fname = head args
    t <- readToTree fname
    let rules = prettyRules $ nub $ map fromJust $ filter isJust $ processTree t
    _ <- mapM putStrLn rules
    return ()
