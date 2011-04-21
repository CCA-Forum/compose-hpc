module Paul.Annotation where

import Text.ParserCombinators.Parsec

type Prefix = String

ident :: Parser String
ident = do
  x <- letter
  xs <- many (alphaNum <|> char '_')
  return (x:xs)

annotation :: String -> Parser (String,String)
annotation prefix = do
  spaces
  string prefix
  spaces
  h <- ident
  space -- Only consume one space character after the AID.
  x <- many anyChar
  return (h,x)
  
recogize :: Prefix -> String -> Maybe (String,String)
recogize prefix s = 
  case parse (annotation prefix) "Annotation" s of
    Left err -> Nothing
    Right x -> Just x
