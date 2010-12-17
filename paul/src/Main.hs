import Paul.Annotation
import Paul.Parser.KeyValuePairs (keyValuePairs)
import Data.Maybe (catMaybes)
import Language.C.Comments
import System.Environment (getArgs)

prefix :: Prefix
prefix = "%%%"

handleAnnotation :: (String,String) -> IO ()
handleAnnotation (h,s) = do
  putStrLn $ "Handle: " ++ h
  case keyValuePairs s of
    Left err -> print err
    Right pairs -> mapM_ print pairs

main :: IO ()
main = do
  [file] <- getArgs
  cs <- comments file
  let cmnts = map commentTextWithoutMarks cs 
  let anns = catMaybes $ map (recogize prefix) cmnts
  mapM_ handleAnnotation anns
  