{-|
  This file contains code related to Graphviz visualization of
  the trees that we work with.  At some point, this may be
  replaced with a proper Graphviz library binding but for now,
  this is sufficient since we aren't making any extensive use
  of graphviz's features other than coloring edges and vertices.
-}

--
--  matt@galois.com // july 2012
--
module RuleGen.Util.Graphviz (
	dumpGraphvizToFile,
	etreeToGraphviz,
	treeToGraphviz,
  wtreeToGraphviz
) where

import System.IO
import RuleGen.Data.Trees
import Data.Tree
import RuleGen.Yang
import RuleGen.Weaver
import RuleGen.Util.IDGen
import Data.List

cleanlabel :: String -> String
cleanlabel lbl = filter (\c -> c /= '\\' && c /= '\'' && c /= '\"') lbl

{-|
  Take a LabeledTree and return a list of lines for the
  corresponding graphviz DOT file.
-}
treeToGraphviz :: LabeledTree -- ^ Tree to print
               -> [String]    -- ^ DOT-file lines
treeToGraphviz t = snd $ evalIDGen t tToGV

{-|
  Take a EditTree and return a list of lines for the
  corresponding graphviz DOT file.
-}
etreeToGraphviz :: EditTree -- ^ Tree to print
                -> [String] -- ^ DOT-file lines
etreeToGraphviz t = snd $ evalIDGen t etToGV 

{-|
  Take a WeaveTree and return a list of lines for the
  corresponding graphviz DOT file.
-}
wtreeToGraphviz :: WeaveTree -- ^ Tree to print
                -> [String]  -- ^ DOT-file lines
wtreeToGraphviz t = snd $ evalIDGen t wToGV

{-|
  IO function to write a sequence of DOT file lines to
  a file.
-}
dumpGraphvizToFile :: String    -- ^ Filename for DOT file
                   -> [String]  -- ^ DOT-file lines
                   -> IO ()
dumpGraphvizToFile fname ls = do
  h <- openFile fname WriteMode
  hPutStrLn h "digraph G {"
  _ <- mapM (\s -> hPutStrLn h ("  "++s)) ls
  hPutStrLn h "}"
  hClose h

--
-- edge makers
--
makeAttrEdge :: Int -> Int -> Maybe [String] -> String
makeAttrEdge i j Nothing = "NODE"++(show i)++" -> NODE"++(show j)++";"
makeAttrEdge i j (Just as) = "NODE"++(show i)++" -> NODE"++(show j)++" ["++a++"];"
  where a = intercalate "," as

-- helper for common case with no attribute : avoid having to write Nothing
-- all over the place
makeEdge :: Int -> Int -> String
makeEdge i j = makeAttrEdge i j Nothing

-- node maker
makeNode :: Int -> [String] -> String -> String
makeNode i attrs lbl =
  "NODE"++(show i)++" ["++a++"];"
  where a = intercalate "," (("label=\""++(cleanlabel lbl)++"\""):attrs)

cGreen :: String
cGreen = "color=green"

cRed :: String
cRed   = "color=red"

cBlue :: String
cBlue  = "color=blue"

cBlack :: String
cBlack = "color=black"

aBold :: String
aBold  = "style=bold"

wpLabel :: WeavePoint -> String
wpLabel (Match _) = "MATCH"
wpLabel (Mismatch _ _) = "MISMATCH"
wpLabel (RightHole _) = "RHOLE"
wpLabel (LeftHole _) = "LHOLE"

wpToGV :: WeavePoint -> IDGen (Int, [String])
wpToGV wp = do
  myID <- genID
  let self = makeNode myID [cGreen] (wpLabel wp)
  case wp of
    Match t -> do (kidID, kidStrings) <- wToGV t
                  let kEdge = makeEdge myID kidID
                  return (myID, self:kEdge:kidStrings)
    Mismatch a b ->  do (kidID1, kidStrings1) <- wToGV a
                        (kidID2, kidStrings2) <- wToGV b
                        let kEdge1 = makeEdge myID kidID1
                            kEdge2 = makeEdge myID kidID2
                        return (myID, self:kEdge1:kEdge2:(kidStrings1++kidStrings2))
    LeftHole t -> do (kidID, kidStrings) <- wToGV t
                     let kEdge = makeEdge myID kidID
                     return (myID, self:kEdge:kidStrings)
    RightHole t -> do (kidID, kidStrings) <- wToGV t
                      let kEdge = makeEdge myID kidID
                      return (myID, self:kEdge:kidStrings)

wToGV :: WeaveTree -> IDGen (Int, [String])
wToGV (WLeaf t) = do
  myID <- genID
  let self = makeNode myID [cGreen] "WLeaf"
  (kidID, kidStrings) <- tToGV t
  let kidEdge = makeEdge myID kidID
  return (myID, self:kidEdge:kidStrings)
wToGV (WNode lbl wps) = do
  myID <- genID
  let self = makeNode myID [cGreen] ("WNode:"++(show lbl))
  processed <- mapM wpToGV wps
  let (kIDs, kSs) = unzip processed
      kidEdges = map (makeEdge myID) kIDs
  return (myID, self:(kidEdges++(concat kSs)))


--
-- node attributes for different node types
--
tToGV :: LabeledTree -> IDGen (Int, [String])
tToGV (Node label kids) = do
  myID <- genID
  let self = makeNode myID [cRed] (show label)
  processedKids <- mapM tToGV kids
  let (kidIDs, kidStrings) = unzip processedKids 
      kidEdges = map (makeEdge myID) kidIDs
  return (myID, self:(kidEdges++(concat kidStrings)))


etToGV :: EditTree -> IDGen (Int,[String])
etToGV (ENil)    = error "etToGV encountered ENil"
etToGV (ELeaf t) = tToGV t
etToGV (ENode label kids) = do
  myID <- genID
  let self = makeNode myID [cBlue] (show label)
      (kidOps, kidTrees) = unzip kids
  processedKids <- mapM etToGV kidTrees
  let kidOperations = map (\i -> case i of
                                   Keep -> [cBlack]
                                   Delete -> [cRed,aBold]) kidOps
  let (kidIDs, kidStrings) = unzip processedKids
      annotatedKidIDs = zip kidIDs kidOperations 
      kidEdges = map (\(j,a) -> makeAttrEdge myID j (Just a)) 
                     annotatedKidIDs
  return (myID, self:(kidEdges++(concat kidStrings)))