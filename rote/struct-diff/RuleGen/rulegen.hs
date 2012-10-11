{-

This is the rule generator.  The goal of this program is to take two programs in
their ATerm representation as provided by minitermite, and produce a set of stratego
rewrite rules that will take the first (the "source" program) and turn it into
the second (the "target" program).

Without any additional work, this mapping represents a rewrite-based patching: the
rules produced will turn the exactly source program into exactly the target, and
the rules will not be generalizable.  Our goal is to produce something more
general, such that the source -> target transformation can be used to derive
general patterns for rewriting.  For example:

x = a*(y+z)  ==>  x = (a * y) + (a * z)

Should represent the distributive law for any variables x, a, y, and z.  Without
generalization, the rule will ONLY apply the distributive law for the term
"x = a*(y+z)", where the variables are named.  

This program is based on the following papers:

 - blah
 - blah

Contact : matt sottile (matt@galois.com)
          geoff hulette (ghulette@gmail.com)

July 2012

-}

import RuleGen.AtermUtilities
import RuleGen.Weaver
import RuleGen.Stratego
import RuleGen.Yang
import RuleGen.Trees
import RuleGen.Pruner
import RuleGen.Contextualizer
import Data.Maybe
import RuleGen.GraphvizUtil
import System.Environment (getArgs)
import System.Console.GetOpt
import System.Exit 
import Data.Tree

despecifyFile :: LabeledTree -> LabeledTree
despecifyFile t = replaceSubtrees "file_info" replacementNode t
  where replacementNode = Node "gen_info()" []

data Flag = Source String
          | Target String
          | GVSource String
          | GVTarget String
          | GVWeave String
          | Output String
          | GVizOnly
          | Debug
  deriving (Show, Eq)

options :: [OptDescr Flag]
options = [
  Option ['d'] ["debug"]           (NoArg Debug)             "Enable debugging output",
  Option ['g'] ["graphviz"]        (NoArg GVizOnly)          "Emit the desired graphviz files and exit immediately",
  Option ['s'] ["source"]          (ReqArg Source "FILE")    "Source file",
  Option ['t'] ["target"]          (ReqArg Target "FILE")    "Target file",
  Option ['S'] ["source-graphviz"] (ReqArg GVSource "FILE")  "Graphviz output for source diff",
  Option ['T'] ["target-graphviz"] (ReqArg GVTarget "FILE")  "Graphviz output for target diff",
  Option ['W'] ["woven-graphviz"]  (ReqArg GVWeave "FILE")   "Graphviz output for woven edit trees",
  Option ['o'] ["output"]          (ReqArg Output "FILE")    "Stratego rule file output"]

header :: String
header = "Usage: rulegen [OPTION...]"

getOutput :: [Flag] -> String
getOutput [] = error $ "stratego rule file required\n" ++ usageInfo header options
getOutput ((Output fname):_) = fname
getOutput (_:rest) = getOutput rest

getSource :: [Flag] -> String
getSource [] = error $ "source file required\n" ++ usageInfo header options
getSource ((Source fname):_) = fname
getSource (_:rest) = getSource rest

getTarget :: [Flag] -> String
getTarget [] = error $ "target file required\n" ++ usageInfo header options
getTarget ((Target fname):_) = fname
getTarget (_:rest) = getTarget rest

getGVSource :: [Flag] -> Maybe String
getGVSource []                   = Nothing
getGVSource ((GVSource fname):_) = Just fname
getGVSource (_:rest)             = getGVSource rest

getGVWeave :: [Flag] -> Maybe String
getGVWeave []                   = Nothing
getGVWeave ((GVWeave fname):_)  = Just fname
getGVWeave (_:rest)             = getGVWeave rest

getGVTarget :: [Flag] -> Maybe String
getGVTarget []                   = Nothing
getGVTarget ((GVTarget fname):_) = Just fname
getGVTarget (_:rest)             = getGVTarget rest

isDebuggingOn :: [Flag] -> Bool
isDebuggingOn []        = False
isDebuggingOn (Debug:_) = True
isDebuggingOn (_:rest)  = isDebuggingOn rest

isGVizOnlyOn :: [Flag] -> Bool
isGVizOnlyOn []           = False
isGVizOnlyOn (GVizOnly:_) = True
isGVizOnlyOn (_:rest)     = isGVizOnlyOn rest

equivalentSet :: [String]
equivalentSet = ["for_statement", "while_stmt"]

labelcompare :: String -> String -> Bool
labelcompare a b 
  | a == b    = True
  | otherwise = if (a `elem` equivalentSet && b `elem` equivalentSet) then True
                                                                      else False

main :: IO ()
main = do
  -- command line argument handling
  args <- getArgs
  let parsedArgs = getOpt RequireOrder options args
      (flags, _, _) = parsedArgs
  case parsedArgs of
    (_ , [],      [])   -> return ()
    (_ , nonOpts, [])   -> error $ "unrecognized arguments: " ++ unwords nonOpts
    (_ , _ ,      msgs) -> error $ concat msgs ++ usageInfo header options

  -- get info from options
  let outputfile = getOutput flags
      sourcefile = getSource flags
      targetfile = getTarget flags
      sgraphviz  = getGVSource flags
      tgraphviz  = getGVTarget flags 
      wgraphviz  = getGVWeave flags
      debugflag  = isDebuggingOn flags
      gvizflag   = isGVizOnlyOn flags

  -- read in trees from term files
  tree1 <- readToTree sourcefile
  tree2 <- readToTree targetfile

  -- clean up trees to remove fileinfo nodes that will induce many false
  -- positive diffs
  let tree1' = despecifyFile tree1
      tree2' = despecifyFile tree2

  -- run Yang's algorithm
      (y1',y2) = treediff tree1' tree2' labelcompare

      y1 = replaceEditTreeNode "gen_info()" (ENode "_" []) (Node "_" []) y1'

  -- check if we want graphviz files dumped of the two diff trees
  case sgraphviz of
    Just fname -> do let g = etreeToGraphviz y1
                     dumpGraphvizToFile fname g
    Nothing    -> return ()
  case tgraphviz of
    Just fname -> do let g = etreeToGraphviz y2
                     dumpGraphvizToFile fname g
    Nothing    -> return ()

  -- create a forest of woven diff trees
  let woven' = weave y1 y2

  -- do we want to dump this out as graphviz?
  case wgraphviz of
    Just fname -> do let g = wtreeToGraphviz woven'
                     dumpGraphvizToFile fname g
    Nothing    -> return ()

  -- check if user only wants the graphviz dumps - exit here if so.
  -- TODO: this still requires the stratego output file to be specified
  --       since it is a required argument.  figure out how to suppress
  --       that if this flag was enabled.
  if gvizflag then do exitSuccess
              else return ()

  --
  -- contextualize : this seeks holes, and produces pairs of pre-transform/
  -- post-transform trees representing the insertion or deletion of code.
  -- the trick here is to pop up a level in the tree to get context of where
  -- the change occurs, and turn siblings into generic stratego variables
  -- NOTE: may not want to do that to siblings if we want the transform to
  --       be for one SPECIFIC place, versus a general pattern.  May
  --       want both available, with some control possible as to which is
  --       chosen and when.
  --  
  let holes = contextualize woven'
      (pre,post) = unzip holes

  if (debugflag) then do putStrLn "---PRE---"
                         _ <- mapM putStrLn (map drawTree pre)
                         putStrLn "---POST---"
                         _ <- mapM putStrLn (map drawTree post)
                         return ()
                 else return ()

  let hole_rules = map (\(a,b) -> (treeToRule a, treeToRule b)) holes

  let nonmatching_forest = nonMatchForest woven'

  -- debug : print stuff out
  if (debugflag) then do putStrLn $ show woven'
                         _ <- mapM (\i -> putStrLn $ show i) nonmatching_forest
                         return ()
                 else return ()

  -- get the rules
  let mismatch_rules = map fromJust $ 
                       filter isJust $ 
                       map toRule nonmatching_forest

      rules = mismatch_rules ++ hole_rules

  -- emit the stratego file
  writeFile outputfile (strategoRules rules)

--  putStrLn (strategoRules rules)
  return ()
