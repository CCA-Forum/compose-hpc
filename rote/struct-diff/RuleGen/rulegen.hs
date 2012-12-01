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

Contact : matt sottile (matt@galois.com)
          geoff hulette (ghulette@gmail.com)
          
-}

import RuleGen.AtermUtilities
import RuleGen.Weaver
import RuleGen.Stratego
import RuleGen.IDGen
import RuleGen.Yang
import RuleGen.Trees
--import RuleGen.Pruner
import RuleGen.Filter
import RuleGen.Contextualizer
import RuleGen.CmdlineArgs
import Data.Maybe
import RuleGen.GraphvizUtil
import System.Exit 
import Data.Tree



-- ==============================================
-- main program
-- ==============================================
main :: IO ()
main = do
  -- command line argument handling
  flags <- handleCommandLine

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

  -- label comparator that is used by the tree diff algorithm
      labelcompare = (==)

  -- run Yang's algorithm
      (y1',y2) = treediff tree1' tree2' labelcompare

      blank = LBLString "_"
      y1 = replaceEditTreeNode (LBLString "gen_info()") (ENode blank []) (Node blank []) y1'

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
  let holes = evalIDGen woven' contextualize
      (pre,post) = unzip holes

  if (debugflag) then do putStrLn "---PRE---"
                         _ <- mapM putStrLn (map dumpTree pre)
                         putStrLn "---POST---"
                         _ <- mapM putStrLn (map dumpTree post)
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
  case rules of
    [] -> do putStrLn "No difference identified."
    _ -> do writeFile outputfile (strategoRules rules)

--  putStrLn (strategoRules rules)
  return ()
