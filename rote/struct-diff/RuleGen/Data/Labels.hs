module RuleGen.Data.Labels (
    Label(..),
    LabelSet,
    toLabelSet,
    toLabelList
) where

import qualified Data.Set as S

data Label = LBLString String
           | LBLList
           | LBLInt Integer
  deriving (Eq, Ord)

instance Show Label where
  show (LBLString s) = s
  show (LBLList)     = "LIST"
  show (LBLInt i)    = show i

type LabelSet = S.Set Label

toLabelSet :: [String] -> LabelSet
toLabelSet lbls = S.fromList $ map LBLString lbls

toLabelList :: [String] -> [Label]
toLabelList = map LBLString

