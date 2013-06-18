module Data.Tree.Labels (
    Label(..),
    LabelSet,
    toLabelSet,
    toLabelList
) where

import qualified Data.Set as S

data Label = LBLString String
           | LBLList
           | LBLInt Integer
  deriving (Read, Show, Eq, Ord)

type LabelSet = S.Set Label

toLabelSet :: [String] -> LabelSet
toLabelSet lbls = S.fromList $ map LBLString lbls

toLabelList :: [String] -> [Label]
toLabelList = map LBLString

