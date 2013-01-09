module RuleGen.IDGen (
    IDGen,
    genID,
    genName,
    evalIDGen
) where

import Control.Monad.State

-- use state monad for generating unique identifiers
type IDGen = State Int

-- generate sequence of unique ID numbers
genID :: IDGen Int
genID = do
  i <- get
  put (i+1)
  return i

-- generate variables with a fixed prefix and a unique suffix.  For
-- example, with prefix "x", a sequence of invocations of this
-- function will yield the names "x0", "x1", "x2", and so on.
genName :: String -> IDGen String
genName n = do
  i <- genID
  return $ n ++ (show i)

evalIDGen :: a -> (a -> IDGen b) -> b
evalIDGen x f = evalState (f x) 0

