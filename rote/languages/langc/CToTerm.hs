import System (getArgs)
import Data.List (intercalate)
import Language.C
import Language.C.Data.Ident
import Language.C.System.GCC (newGCC)

data Term = Term String [Term]

instance Show Term where
  show (Term x []) = x
  show (Term x xs) = x ++ " [" ++ (join (map show xs)) ++ "]"
    where join = intercalate " "

class Termable a where
  term :: a -> Term

instance Termable a => Termable (Maybe a) where
  term Nothing = Term "Nothing" []
  term (Just x) = Term "Just" [term x]

-- Convert AST to a term

instance Termable CTranslUnit where
  term (CTranslUnit decls _) = 
    Term "Unit" (map term decls)

instance Termable CExtDecl where
  -- function definition
  term (CFDefExt funDef) = term funDef
  -- global declaration, undefined until we need it
  term (CDeclExt _)      = undefined
  -- assembly stuff, unsupported, will be error
  term (CAsmExt _)       = error "Asm in C code unsupported"

instance Termable CFunDef where
  term (CFunDef specifiers (CDeclr (Just (funcId)) _ _ _ _) decls stmt _) =
    Term "Func" [ term funcId,
                  Term "FunSpecifiers" $ map term specifiers,
                  Term "Body" [term stmt]]

instance Termable CDeclSpec where
  term (CTypeSpec typeSpec) = term typeSpec
  term (CStorageSpec storSpec) = term storSpec
  term (CTypeQual typeQual) = term typeQual

instance Termable CTypeSpec where
  term (CVoidType _) = Term "Void" []
  term (CCharType _) = Term "Char" []
  term (CShortType _) = Term "Short" []
  term (CIntType _) = Term "Int" []
  term (CLongType _) = Term "Long" []
  term (CFloatType _) = Term "Float" []
  term (CDoubleType _) = Term "Double" []
  term (CSignedType _) = Term "Signed" []
  term (CUnsigType _) = Term "Unsigned" []
  term (CBoolType _) = Term "Bool" []
  term _ = error "CTypeSpec"

instance Termable CDecl where
 term (CDecl ds [(Just x, _, _)] _) = 
   Term "Declaration" [ Term "DSpec" $ map term ds,
                        term x ]
 term _ = error "CDecl"

instance Termable CDeclr where
  term (CDeclr ident derivedDeclrs mbStrLit _ _) =
    Term "Declr" [term ident,Term "Derived" (map term derivedDeclrs)]
    
instance Termable Ident where
 term (Ident x _ _) = Term ("(Ident " ++ (show x) ++ ")") []

declIdent :: CDecl -> Ident
declIdent (CDecl _ [(Just (CDeclr (Just x) _ _ _ _), _, _)] _) = x
declIdent _ = error "declrIdent"

instance Termable CDerivedDeclr where
  term (CPtrDeclr typeQuals _) = Term "CPtr" []
  term (CArrDeclr typeQuals arrSize _) = error "CDerivedDeclr"
  term (CFunDeclr (Left idents) attrs _) = error "CDerivedDeclr"
  term (CFunDeclr (Right (decls,isVariadic)) attrs _) = 
    let argIds = map declIdent decls in
    Term "FuncDeclr" [Term "Params" (map term argIds)]

instance Termable CStorageSpec where
  term (CAuto _) = Term "Auto" []
  term (CRegister _) = Term "Register" []
  term (CStatic _) = Term "Static" []
  term (CExtern _) = Term "Extern" []
  term (CTypedef _) = Term "Typedef" []
  term (CThread _) = error "Unsupported GNUC extension"

instance Termable CTypeQual where
  term (CConstQual _) = Term "Const" []
  term (CVolatQual _) = Term "Volatile" []
  term (CRestrQual _) = Term "Restrict" []
  term (CInlineQual _) = Term "Inline" []
  term (CAttrQual _) = error "attribute type qualifiers unsupported"

instance Show CTypeQual where
  show (CConstQual _) = "const"
  show (CVolatQual _) = "volatile"
  show (CRestrQual _) = "restrict"
  show (CInlineQual _) = "inline"
  show (CAttrQual _) = "const"

instance Termable CStat where
  term (CExpr (Just expr) _) = 
    Term "ExprStmt" [term expr]
  term (CCompound idents blockItems _) = 
    Term "CompoundStmt" (map term blockItems)
  term (CReturn mbExpr _) =
    Term "ReturnStmt" [term mbExpr]

instance Termable CBlockItem where
  term (CBlockStmt stat) = term stat
  term (CBlockDecl decl) = term decl

instance Termable CExpr where
  term (CAssign CAssignOp expr1 expr2 _) = 
    Term "Assign" [term expr1,term expr2]
  term (CCast decl expr _) = 
    Term "Cast" [term decl,term expr]
  term (CCall expr argExprs _) = 
    Term "Call" [term expr,Term "Args" (map term argExprs)]
  term (CVar ident _) = 
    Term "Var" [term ident]
  term (CConst cnst) = term cnst
  term _ = error "CExpr"

constTermStr :: Show a => a -> String
constTermStr x = "(Const " ++ show x ++ ")"

instance Termable CConst where
  term (CIntConst n _) = Term (constTermStr n) []
  term (CCharConst c _) = Term (constTermStr c) []
  term (CFloatConst f _) = Term (constTermStr f) []
  term (CStrConst s _) = Term (constTermStr s) []

main :: IO ()
main = do
  [file] <- getArgs
  result <- parseCFile (newGCC "gcc") Nothing [] file
  case result of
    Left err -> print err
    Right x -> print (term x)
