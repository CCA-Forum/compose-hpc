import Data.List (intercalate)
import Language.C
import Language.C.Data.Ident
import Language.C.System.GCC (newGCC)

data Term = Term String [Term]

instance Show Term where
  show (Term x xs) = "(" ++ (join (x : map show xs)) ++ ")"
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
  term (CFDefExt funDef) = term funDef

instance Termable CFunDef where
  term (CFunDef [retType] (CDeclr (Just (funcId)) _ _ _ _) decls stmt _) =
    Term "Func" [ term funcId,
                  term retType,
                  Term "Body" [term stmt]]

instance Termable CDeclSpec where
  term (CTypeSpec typeSpec) = term typeSpec
  term (CStorageSpec storSpec) = error "CDeclSpec"
  term (CTypeQual typeQual) = error "CDeclSpec"

instance Termable CTypeSpec where
  term (CVoidType _) = Term "void" []
  term (CCharType _) = Term "char" []
  term (CShortType _) = Term "short" []
  term (CIntType _) = Term "int" []
  term (CLongType _) = Term "long" []
  term (CFloatType _) = Term "float" []
  term (CDoubleType _) = Term "double" []
  term (CSignedType _) = Term "signed" []
  term (CUnsigType _) = Term "unsigned" []
  term (CBoolType _) = Term "bool" []
  term _ = error "CTypeSpec"

instance Termable CDecl where
 term (CDecl _ [(Just x, _, _)] _) = term x
 term _ = error "CDecl"

instance Termable CDeclr where
  term (CDeclr ident derivedDeclrs mbStrLit _ _) =
    Term "Declr" [term ident,Term "Derived" (map term derivedDeclrs)]
    
instance Termable Ident where
 term (Ident x _ _) = Term "Ident" [Term (show x) []]

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

instance Termable CConst where
  term (CIntConst cint _) = Term "Const" [Term (show cint) []]
  term (CCharConst cchar _) = Term "Const" [Term (show cchar) []]
  term (CFloatConst cfloat _) = Term "Const" [Term (show cfloat) []]
  term (CStrConst cstring _) = Term "Const" [Term (show cstring) []]

main :: IO ()
main = do
  let file = "examples/guard.c"
  result <- parseCFile (newGCC "gcc") Nothing [] file
  case result of
    Left err -> print err
    Right x -> do
      print (pretty x)
      putStrLn "----"
      print (term x)
