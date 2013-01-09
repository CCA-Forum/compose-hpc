module RuleGen.CmdlineArgs (
  Flag,
  handleCommandLine,
  getOutput,
  getSource,
  getTarget,
  getGVSource,
  getGVTarget,
  getGVWeave,
  isDebuggingOn,
  isGVizOnlyOn
) where

import System.Environment (getArgs)
import System.Console.GetOpt

handleCommandLine :: IO [Flag]
handleCommandLine = do
  args <- getArgs
  let parsedArgs = getOpt RequireOrder options args
      (flags, _, _) = parsedArgs
  case parsedArgs of
    (_ , [],      [])   -> return flags
    (_ , nonOpts, [])   -> error $ "unrecognized arguments: " ++ unwords nonOpts
    (_ , _ ,      msgs) -> error $ concat msgs ++ usageInfo header options


-- ==============================================
-- command line argument handling
-- ==============================================
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
  Option ['g'] ["graphviz"]        (NoArg GVizOnly)          "Emit the desired graphviz files and exit",
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