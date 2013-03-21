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
import RuleGen.Data.Labels
import RuleGen.Util.Misc

data ConfigurationRule =
    PreFilter PreFilterRule
  | PostFilter PostFilterRule
  | GeneralizeFilter GeneralizeFilterRule
  | ContextualizeFilter ContextualizeFilterRule
  deriving (Show, Eq)

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
