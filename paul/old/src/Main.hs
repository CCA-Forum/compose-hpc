import Paul.Annotation
import Paul.Parser.KeyValuePairs (keyValuePairs)
import Data.Maybe (catMaybes)
import Language.C.Comments
import Language.C
import Language.C.System.GCC
import System.Environment (getArgs)

prefix :: Prefix
prefix = "%%%"

lineNum :: CNode a => a -> Int
lineNum = posRow . posOfNode . nodeInfo

handleAnnotation :: CNode a => a -> (String,String) -> IO ()
handleAnnotation ast (h,s) = do
  putStrLn $ "Handle: " ++ h
  case keyValuePairs s of
    Left err -> print err
    Right pairs -> mapM_ print pairs

parseAST :: FilePath -> IO CTranslUnit
parseAST input_file = do 
  parse_result <- parseCFile (newGCC "gcc") Nothing [] input_file
  case parse_result of
    Left parse_err -> error (show parse_err)
    Right ast      -> return ast

main :: IO ()
main = do
  [file] <- getArgs
  cs <- comments file
  ast <- parseAST file
  let cmnts = map commentTextWithoutMarks cs
  let anns = catMaybes $ map (recogize prefix) cmnts
  mapM_ (handleAnnotation ast) anns
