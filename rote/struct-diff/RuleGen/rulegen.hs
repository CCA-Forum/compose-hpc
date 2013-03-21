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

import RuleGen.Generalize
import RuleGen.Data.Aterm
import RuleGen.Data.Trees
import RuleGen.Weaver
import RuleGen.Stratego
import RuleGen.Util.IDGen
import RuleGen.Yang
--import RuleGen.Pruner
import RuleGen.Filter
import RuleGen.Contextualizer
import RuleGen.Util.CmdlineArgs
import RuleGen.Util.Graphviz
import RuleGen.Util.Configuration
import System.Exit 
import Data.Tree
import Control.Monad (when)
import Data.Maybe (mapMaybe)


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
      configfile = getConfigFile flags

  -- read configuration rules into a big rule list
  configRules <- case configfile of
                   Nothing    -> return []
                   Just fname -> readConfig fname

  --
  -- partition configuration rules into the four sets that
  -- correspond to preFilter, postFilter, generalizeFilter, and
  -- contextualizeFilter phases.
  --
  let (preFilt, postFilt, gFilt, cFilt) = partitionRuleSets configRules

  when debugflag $ do
    putStrLn $ "PREFILT: "++(show preFilt)
    putStrLn $ "POSTFILT: "++(show postFilt)
    putStrLn $ "GFILT: "++(show gFilt)
    putStrLn $ "CFILT: "++(show cFilt)

  -- read in trees from term files
  tree1 <- readToTree sourcefile
  tree2 <- readToTree targetfile

  -- clean up trees to remove fileinfo nodes that will induce many false
  -- positive diffs
  let tree1' = preFilters preFilt tree1
      tree2' = preFilters preFilt tree2

      -- label comparator that is used by the tree diff algorithm
      labelcompare = (==)

      -- run Yang's algorithm
      (y1',y2) = treediff tree1' tree2' labelcompare

      --
      -- TODO: the following commented out replacement of gen_info() subtrees in
      -- the y1 tree is necessary to allow us to pattern match on actual files
      -- with file_info() subtrees, but it screws up the generalization performed
      -- later on.  Perhaps we should do a replaceWeaveTreeNode to perform this cleanup of
      -- gen_info() subtrees after we know we won't manipulate the trees any more.  for now,
      -- don't modify y1 so just let y1 = y1'.
      --

      --blank = LBLString "_"
      --y1 = replaceEditTreeNode (LBLString "gen_info()") (ENode blank []) (Node blank []) y1'
      y1 = y1'

  -- graphviz files dumped of the two diff trees (or not if the filename is
  -- Nothing).
  dumpGraphvizToFile sgraphviz (etreeToGraphviz y1)
  dumpGraphvizToFile tgraphviz (etreeToGraphviz y2)

  -- create a forest of woven diff trees
  let woven' = weave y1 y2 False

  -- dump weave tree as graphviz is wgraphviz isn't Nothing
  dumpGraphvizToFile wgraphviz (wtreeToGraphviz woven')

  -- check if user only wants the graphviz dumps - exit here if so.
  -- TODO: this still requires the stratego output file to be specified
  --       since it is a required argument.  figure out how to suppress
  --       that if this flag was enabled.
  when gvizflag exitSuccess

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
  let holes = evalIDGen woven' (contextualize cFilt)
      (pre,post) = unzip holes

  when debugflag $ do
    putStrLn "---PRE---"
    mapM_ putStrLn (map dumpTree pre)
    putStrLn "---POST---"
    mapM_ putStrLn (map dumpTree post)

  let hole_rules = map (\(a,b) -> (treeToRule a, treeToRule b)) holes

  let nonmatching_forestPre = nonMatchForest woven'
      nonmatching_forest = map (generalizeWeave gFilt) nonmatching_forestPre
      --nonmatching_forest = nonmatching_forestPre

  -- debug : print stuff out
  when debugflag $ do
    print woven'
    mapM_ print nonmatching_forest

  -- get the rules
  let blank = LBLString "_"
      nonmatching_forest' = map (\p -> replaceWeavePoint (LBLString "gen_info()") 
                                                         (WNode blank False [])
                                                         (Node blank []) 
                                                         (True,False) 
                                                         p) 
                                nonmatching_forest
      mismatch_rules = mapMaybe toRule nonmatching_forest'
      rules = mismatch_rules ++ hole_rules

  -- emit the stratego file
  if null rules
    then putStrLn "No difference identified."
    else writeFile outputfile (strategoRules rules)

  return ()
