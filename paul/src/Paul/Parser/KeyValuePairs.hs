module Paul.Parser.KeyValuePairs
( keyValuePairs
, Value (..)
) where

import Text.ParserCombinators.Parsec
import qualified Text.ParserCombinators.Parsec.Token as P
import Text.ParserCombinators.Parsec.Language (emptyDef)
import Data.Char

type Key = String

data Value = StringValue String
           | BoolValue Bool
           | IntValue Integer
           | FloatValue Double
           | IdentifierValue String
           deriving (Eq,Show)

-- Because we are using lexeme parsers, we have to be careful about parsing
-- whitespace.  In particular, each "token" should consume any trailing
-- whitespace.

lexer = P.makeTokenParser emptyDef
natOrFloat = P.naturalOrFloat lexer
stringLiteral = P.stringLiteral lexer

identifier :: Parser String
identifier = do
  x <- letter
  xs <- many alphaNum
  spaces
  return (x:xs)

wordValue :: Parser Value
wordValue = do
  word <- identifier
  return $ case (map toLower) word of
    "true" -> BoolValue True
    "false" -> BoolValue False
    otherwise -> IdentifierValue word

numValue :: Parser Value
numValue = do
  sign <- option '+' (oneOf "+-")
  num <- natOrFloat
  return $ case num of
    Left nat -> IntValue (applySign sign nat)
    Right dec -> FloatValue (applySign sign dec)
  where
    applySign '+' = id
    applySign '-' = negate

hereDoc :: Parser String
hereDoc = do
  string "<<"
  delimit <- identifier
  txt <- manyTill anyChar $ try (newline >> string delimit)
  spaces
  return txt

stringValue :: Parser Value
stringValue = do
  s <- stringLiteral <|> hereDoc
  return $ StringValue s

value :: Parser Value
value = wordValue <|> stringValue <|> numValue <?> "value"

equals :: Parser ()
equals = char '=' >> spaces

pair :: Parser (Key,Value)
pair = do
  x <- identifier
  equals
  y <- value
  return (x,y)

parsePairs :: Parser [(Key,Value)]
parsePairs = do
  spaces
  pairs <- many pair
  eof
  return pairs

keyValuePairs :: String -> Either ParseError [(Key,Value)]
keyValuePairs = parse parsePairs "KeyValuePairs"
