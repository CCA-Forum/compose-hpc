import Text.ParserCombinators.Parsec
import qualified Text.ParserCombinators.Parsec.Token as Tok
import Text.ParserCombinators.Parsec.Language (haskellStyle)
import Data.List (intercalate)

data Term = Term String [Term]
          | Ident String
          | Const Int
          | Bang Term

instance Show Term where
  show (Term x []) = x
  show (Term x xs) = x ++ " [" ++ (join (map show xs)) ++ "]"
    where join = intercalate " "
  show (Ident x) = "(Ident " ++ show x ++ ")"
  show (Const n) = "(Const " ++ show n ++ ")"
  show (Bang t) = "(! " ++ show t ++ ")"

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
  x <- term
  return (Bang x)

ident :: Parser Term
ident = do
  reserved "Ident"
  x <- stringLit
  return (Ident x)

constInt :: Parser Term
constInt = do
  reserved "Const"
  x <- natural
  return (Const (fromIntegral x))

term :: Parser Term
term =  parens term
    <|> bang
    <|> try ident
    <|> try constInt
    <|> regularTerm
    <?> "Term"

parseTerm :: String -> Either ParseError Term
parseTerm = parse (allOf term) "Term"

termToC :: Term -> String
termToC (Term "Unit" funs) = intercalate "\n" (map termToC funs)
termToC (Term "Func" [(Ident x),(Term "FunSpecifiers" [typ]),(Term "Body" stmts)]) =
  let typStr = termToC typ
      bodyStr = intercalate "\n" (map termToC stmts)
   in typStr ++ " " ++ x ++ " () {\n" ++ bodyStr ++ "}"
termToC (Term "CompoundStmt" stmts) =
  concatMap termToC stmts
termToC (Term "Declaration" [Term "DSpec" typs, Term "Declr" [Term "Just" [Ident x], tqual]]) =
  let typStr = intercalate " " (map termToC typs)
      tqstr = termToC tqual
  in typStr ++ tqstr ++ " " ++ x ++ ";\n"
termToC (Term "Derived" []) = ""
termToC (Term "Derived" [Term "CPtr" []]) = "*"
termToC (Term "ExprStmt" [stmt]) = 
  let stmtStr = termToC stmt
   in stmtStr ++ ";\n"
termToC (Term "Assign" [Term "Var" [Ident x],expr]) =
  let rhs = termToC expr
   in x ++ " = " ++ rhs
termToC (Term "Call" [Term "Var" [Ident x],Term "Args" args]) =
  let argsStr = intercalate "," (map termToC args)
   in x ++ "(" ++ argsStr ++ ")"
termToC (Term "IfStmt" [test,consq]) =
  let testStr = termToC test
      consqStr = termToC consq
   in "if(" ++ testStr ++ "){\n" ++ consqStr ++ ";\n}\n"
termToC (Term "Eq" [lhs,rhs]) = (termToC lhs) ++ " == " ++ (termToC rhs)
termToC (Term "Var" [Ident x]) = x
termToC (Term "Null" []) = "NULL"
termToC (Term "ReturnStmt" [Term "Just" [expr]]) =
  let exprStr = termToC expr
   in "return(" ++ exprStr ++ ");\n"
termToC (Const n) = show n
termToC (Bang t) = termToC t -- Ignore bangs
termToC (Term "Int" []) = "int"
termToC (Term "Double" []) = "double"
termToC (Term "Long" []) = "long"
termToC (Term "Short" []) = "short"
termToC (Term "Float" []) = "float"
termToC (Term "Char" []) = "char"
termToC (Term "Unsigned" []) = "unsigned"
termToC t = error ("Currently unsupported term " ++ (show t))

main :: IO ()
main = do
  ctnts <- getContents
  case parseTerm ctnts of
    Left err -> print err
    Right x -> putStrLn (termToC x)
  
