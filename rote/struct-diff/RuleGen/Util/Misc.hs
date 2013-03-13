{-| A collection of utility functions that are useful in
    general and don't belong embedded in specific modules
    that serve a specific role.
 -}
module RuleGen.Util.Misc (
    forceEither,
    wordsWhen
) where

-- taken from MissingH.  this is so simple, I don't want to induce a
-- package dependency simply for this little function.
forceEither :: Show e => Either e a -> a
forceEither (Left x) = error (show x)
forceEither (Right x) = x

-- see: http://stackoverflow.com/questions/4978578/how-to-split-a-string-in-haskell
wordsWhen     :: (Char -> Bool) -> String -> [String]
wordsWhen p s =  case dropWhile p s of
                      "" -> []
                      s' -> w : wordsWhen p s''
                            where (w, s'') = break p s'