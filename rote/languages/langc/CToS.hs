import Data.List (intercalate)
import Language.C
import Language.C.Data.Ident
import Language.C.System.GCC (newGCC)

data Term = Term String [Term]

instance Show Term where
  show (Term x []) = x
  show (Term x xs) = "(" ++ x ++ " " ++ (join (map show xs)) ++ ")"
    where join = intercalate " "

termMap :: (a -> Term) -> [a] -> Term
termMap _ [] = Term "nil" []
termMap f xs = Term "lst" (map f xs)

termMaybe :: (a -> Term) -> Maybe a -> Term
termMaybe f Nothing = Term "nothing" []
termMaybe f (Just x) = Term "just" [f x]


-- Convert AST to a term

cTranslUnitToTerm :: CTranslUnit -> Term
cTranslUnitToTerm (CTranslUnit decls _) = 
  Term "CTranslUnit" [(termMap cExtDeclToTerm) decls]

cExtDeclToTerm :: CExtDecl -> Term
cExtDeclToTerm (CFDefExt funDef) = 
  Term "CFDefExt" [cFunDefToTerm funDef]

cFunDefToTerm :: CFunDef -> Term
cFunDefToTerm (CFunDef declSpecs declr decls stat _) =
  Term "CFunDef" [(termMap cDeclSpecToTerm) declSpecs,
                  cDeclrToTerm declr,
                  (termMap cDeclToTerm) decls,
                  cStatToTerm stat]

cDeclSpecToTerm :: CDeclSpec -> Term
cDeclSpecToTerm (CStorageSpec storSpec) = error "cDeclSpecToTerm"
cDeclSpecToTerm (CTypeSpec typeSpec) = 
  Term "CTypeSpec" [cTypeSpecToTerm typeSpec]
cDeclSpecToTerm (CTypeQual typeQual) = error "cDeclSpecToTerm"

cTypeSpecToTerm :: CTypeSpec -> Term
cTypeSpecToTerm (CVoidType _) = Term "CVoidType" []
cTypeSpecToTerm (CCharType _) = Term "CCharType" []
cTypeSpecToTerm (CShortType _) = Term "CShortType" []
cTypeSpecToTerm (CIntType _) = Term "CIntType" []
cTypeSpecToTerm (CLongType _) = Term "CLongType" []
cTypeSpecToTerm (CFloatType _) = Term "CFloatType" []
cTypeSpecToTerm (CDoubleType _) = Term "CDoubleType" []
cTypeSpecToTerm (CSignedType _) = Term "CSignedType" []
cTypeSpecToTerm (CUnsigType _) = Term "CUnsigType" []
cTypeSpecToTerm (CBoolType _) = Term "CBoolType" []
cTypeSpecToTerm _ = error "cTypeSpecToTerm"

cDeclToTerm :: CDecl -> Term
cDeclToTerm (CDecl _ [(Just x, _, _)] _) =
  Term "CDecl" [cDeclrToTerm x]

cDeclrToTerm :: CDeclr -> Term
cDeclrToTerm (CDeclr mbIdent derivedDeclrs mbStrLit attrs _) =
  Term "CDeclr" [(termMaybe identToTerm) mbIdent,
                 (termMap cDerivedDeclrs) derivedDeclrs,
                 (termMaybe cStrLitToTerm) mbStrLit,
                 (termMap cAttrToTerm) attrs]

identToTerm :: Ident -> Term
identToTerm (Ident x _ _) = Term "Ident" [Term x []]

cDerivedDeclrs :: CDerivedDeclr -> Term
cDerivedDeclrs (CPtrDeclr typeQuals _) = 
  Term "CPtrDeclr" [(termMap cTypeQualToTerm) typeQuals]
cDerivedDeclrs (CArrDeclr typeQuals arrSize _) = error "cDerivedDeclrs"
cDerivedDeclrs (CFunDeclr (Left idents) attrs _) = error "cDerivedDeclrs"
cDerivedDeclrs (CFunDeclr (Right (decls,variadic)) attrs _) = 
  Term "CFunDeclr" [(termMap cDeclToTerm) decls]

cStrLitToTerm :: CStrLit -> Term
cStrLitToTerm = undefined

cAttrToTerm :: CAttr -> Term
cAttrToTerm = undefined

cStatToTerm :: CStat -> Term
cStatToTerm (CExpr mbExpr _) = 
  Term "CExpr" [(termMaybe cExprToTerm) mbExpr]
cStatToTerm (CCompound idents blockItems _) = 
  Term "CCompound" [(termMap identToTerm) idents,
                    (termMap cBlockItemToTerm) blockItems]
cStatToTerm (CReturn mbExpr _) =
  Term "CReturn" [(termMaybe cExprToTerm) mbExpr]
  

cBlockItemToTerm :: CBlockItem -> Term
cBlockItemToTerm (CBlockStmt stat) = cStatToTerm stat
cBlockItemToTerm (CBlockDecl decl) = cDeclToTerm decl

cExprToTerm :: CExpr -> Term
cExprToTerm (CAssign CAssignOp expr1 expr2 _) = 
  Term "CAssign" [Term "CAssignOp" [],
                  cExprToTerm expr1,
                  cExprToTerm expr2]
cExprToTerm (CCast decl expr _) = 
  Term "CCast" [cDeclToTerm decl,
                cExprToTerm expr]
cExprToTerm (CCall expr argExprs _) = 
  Term "CCall" [cExprToTerm expr,
                (termMap cExprToTerm) argExprs]
cExprToTerm (CVar ident _) = 
  Term "CVar" [identToTerm ident]
cExprToTerm (CConst cnst) = 
  Term "CConst" [cConstToTerm cnst]
cExprToTerm _ = error "cExprToTerm"

cConstToTerm :: CConst -> Term
cConstToTerm (CIntConst cint _) = 
  Term "CIntConst" [Term (show cint) []]
cConstToTerm (CCharConst cchar _) = 
  Term "CCharConst" [Term (show cchar) []]
cConstToTerm (CFloatConst cfloat _) =
  Term "CFloatConst" [Term (show cfloat) []]
cConstToTerm (CStrConst cstring _) =
  Term "CStrConst" [Term (show cstring) []]

--

main :: IO ()
main = do
  result <- parseCFile (newGCC "gcc") Nothing [] "test.c"
  case result of
    Left err -> print err
    Right x -> do
      print (pretty x)
      putStrLn "----"
      print (cTranslUnitToTerm x)
