{-|
  This module contains code for reading configuration files.  It uses the
  format supported by Data.ConfigFile, which is fairly standard and similar to
  that used by Windows .INI files.  The supported format is based on a config
  file partitioned into a set of sections like this:

  [section_name]
  key=value
  key=value
  key=value

  [other_section_name]
  key=value
  key=value

  and so on...  Section names are arbitary and not interpreted - they simply
  serve to name the blocks of key-value pairs that are meaningful.  The only
  constraint is that section names must be unique.  For a given section name,
  the first key that is interpreted is "phase".  This can take on values
  "pre", "post", "generalize", or "context".  These correspond to the
  different phases in the rule generator.  Pre-filters are substitution
  filters applied before anything happens.  Post-filters are substitution
  filters applied after everything is done and right before rule emission.
  Generalize filters are used to indicate where and how generalization occurs.
  Contextualize filters are used to indicate where holes (missing code on
  one side or the other) are dealt with in the sense of rule generation.
  The remaining key/value pairs in a section depend on the phase.

  Phase: pre
    match=val1, val2, ... , valN :: seek nodes with one of these labels
    substitute=val               :: replace the subtrees that match the labels
                                    above with a node with no children labeled
                                    with this value.

  Phase: post
  (same as pre)

  Phase: generalize
    match=val1, val2, ..., valN  :: seek nodes with one of these
    target=val1, val2, ..., valM :: generalize one of these under a match 

  Phase: context
    match=val1, val2, ..., valN  :: seek nodes with one of these
-}
module RuleGen.Util.Configuration (
  ConfigurationRule (..),
  PreFilterRule (..),
  PostFilterRule (..),
  GeneralizeFilterRule (..),
  ContextualizeFilterRule (..),
  readConfig,
  partitionRuleSets
) where

import Data.ConfigFile
import Data.Tree.Labels
import RuleGen.Util.Misc

{-|
  Single data type representing any type of rule.  All rules come in
  one big mixed up config file, so we use this to hold them all
  until we partition them into classes later.
-}
data ConfigurationRule =
    PreFilter PreFilterRule
  | PostFilter PostFilterRule
  | GeneralizeFilter GeneralizeFilterRule
  | ContextualizeFilter ContextualizeFilterRule
  deriving (Show, Eq)

{-|
  Given a configuration rule set, partition it into four sets for each of
  the legal rule types.  The type checker will make sure only valid members
  occur in each set.
-}
partitionRuleSets :: [ConfigurationRule] -> 
                     ([PreFilterRule], [PostFilterRule], 
                      [GeneralizeFilterRule], [ContextualizeFilterRule])
partitionRuleSets []  = ([],[],[],[])
partitionRuleSets cfg =
  case r of
    (PreFilter f)           -> ((f:prs),pos,gs,cs)
    (PostFilter f)          -> (prs,(f:pos),gs,cs)
    (GeneralizeFilter f)    -> (prs,pos,(f:gs),cs)
    (ContextualizeFilter f) -> (prs,pos,gs,(f:cs))
  where
    (r:rs) = cfg
    (prs,pos,gs,cs) = partitionRuleSets rs

data PreFilterRule = PreFilterRule LabelSet Label
  deriving (Show, Eq)

data PostFilterRule = PostFilterRule LabelSet Label
  deriving (Show, Eq)

data GeneralizeFilterRule = GeneralizeFilterRule LabelSet [Label]
  deriving (Show, Eq)

data ContextualizeFilterRule = ContextualizeFilterRule LabelSet
  deriving (Show, Eq)

{-|
  Read config file.  This does not gracefully deal with
  error conditions.  

  TODO: make graceful with respect to error conditions
-}
readConfig :: String -> IO [ConfigurationRule]
readConfig fname = do
    val <- readfile emptyCP fname
    let cp = forceEither val
        sects = sections cp
        csvs s = map (filter (/=' ')) $ wordsWhen (==',') s
        handleSection s = 
          let phase = head $ csvs $ forceEither $ get cp s "phase"
              match = csvs $ forceEither $ get cp s "match"
              targets = csvs $ forceEither $ get cp s "target"
              substitute = csvs $ forceEither $ get cp s "substitute"
          in case phase of
                "pre"  -> PreFilter $ PreFilterRule (toLabelSet match) 
                                                    (head $ toLabelList substitute)
                "generalize" -> GeneralizeFilter $
                                GeneralizeFilterRule (toLabelSet match) 
                                                     (toLabelList targets)
                "context" -> ContextualizeFilter $ 
                             ContextualizeFilterRule (toLabelSet match)
                "post" -> PostFilter $ PostFilterRule (toLabelSet match)
                                                      (head $ toLabelList substitute)
                _      -> error $ "Phase `"++phase++"' unknown."
    return $ map handleSection sects
