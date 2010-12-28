import Text.ParserCombinators.Parsec
import qualified Text.ParserCombinators.Parsec.Token as Tok
import Text.ParserCombinators.Parsec.Language (haskellStyle)
import Data.List (intercalate)

data Term = Term String [Term]
          | Ident String
          | Const Int

instance Show Term where
  show (Term x []) = x
  show (Term x xs) = x ++ " [" ++ (join (map show xs)) ++ "]"
    where join = intercalate " "
  show (Ident x) = "(Ident " ++ show x ++ ")"
  show (Const n) = "(Const " ++ show n ++ ")"

-- Tokens

lexer :: Tok.TokenParser ()
lexer = Tok.makeTokenParser style
  where names = ["Ident"]
        style = haskellStyle {Tok.reservedNames = names}

lexeme :: Parser a -> Parser a
lexeme = Tok.lexeme lexer

reserved :: String -> Parser ()
reserved = Tok.reserved lexer

natural :: Parser Integer
natural = Tok.natural lexer

parens :: Parser a -> Parser a
parens = Tok.parens lexer

brackets :: Parser a -> Parser a
brackets = Tok.brackets lexer

stringLit :: Parser String
stringLit = Tok.stringLiteral lexer

allOf :: Parser a -> Parser a
allOf p = do
  Tok.whiteSpace lexer
  r <- p
  eof
  return r

-- Terms and rule literals

atom :: Parser String
atom = do
  x <- letter
  xs <- many alphaNum
  return (x:xs)

regularTerm :: Parser Term
regularTerm = do
  x <- lexeme atom
  cs <- option [] (brackets (many term))
  return (Term x cs)

bang :: Parser Term
bang = do
  lexeme (char '!')
  term

ident :: Parser Term
ident = parens $ do
  reserved "Ident"
  x <- stringLit
  return (Ident x)

constInt :: Parser Term
constInt = parens $ do
  reserved "Const"
  x <- natural
  return (Const (fromIntegral x))

term :: Parser Term
term =  bang
    <|> try ident
    <|> try constInt
    <|> regularTerm
    <?> "Term"

parseTerm :: String -> Either ParseError Term
parseTerm = parse (allOf term) "Term"

main :: IO ()
main = do
  ctnts <- getContents
  case parseTerm ctnts of
    Left err -> print err
    Right x -> print x
  
