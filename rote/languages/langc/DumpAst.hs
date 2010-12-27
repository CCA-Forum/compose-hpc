import Data.List (intercalate)
import Language.C
import Language.C.Data.Ident
import Language.C.System.GCC (newGCC)

-- Remember to cabal install genericserialize
import Data.Generics.Serialization.Streams
import Data.Generics.Serialization.SExp

main :: IO ()
main = do
  result <- parseCFile (newGCC "gcc") Nothing [] "examples/guard.c"
  case result of
    Left err -> print err
    Right x -> do
      print (pretty x)
      putStrLn "----"
      putStrLn $ buildList (sexpSerialize x)
      
