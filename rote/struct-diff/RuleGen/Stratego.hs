{-|
  Code to render rewrite rules in the form of StTrees into
  Stratego/XT compatible rule statements.
-}
module RuleGen.Stratego where

import Data.Tree
import Data.List (intercalate)
import Control.Monad.State
--import Debug.Trace

data StLabel = StLit String
             | StLocalVar String
             | StGlobalVar String

type StTree = Tree StLabel

type IDGen = State ([String], [(String,String)])

-- helper to generate IDs
genID :: String -> IDGen String
genID k = do
  (supply,env) <- get
  case lookup k env of
    Just x -> return x
    Nothing -> do let x:supply' = supply
                  let env' = (k,x):env
                  put (supply',env')
                  return x

strategoRules' :: [(StTree,StTree)] -> [String]
strategoRules' ts = evalState (mapM renderRule ts) (symbols,[])
  where 
    symbols = map (\i -> "gen" ++ (show i)) [(1::Int) ..]

renderRule :: (StTree,StTree) -> IDGen String
renderRule (t1,t2) = do
  s1 <- renderStTree t1
  s2 <- renderStTree t2
  return $ s1 ++ " -> " ++ s2

renderStTree :: StTree -> IDGen String
renderStTree (Node lbl []) = renderLabel lbl
renderStTree (Node lbl ks) = do
  s <- renderLabel lbl
  case ks of
    [] -> return s
    _  -> do ss <- mapM renderStTree ks
             return $ s ++ "(" ++ (intercalate "," ss) ++ ")"

renderLabel :: StLabel -> IDGen String
renderLabel (StLit lit) = return lit
renderLabel (StLocalVar loc) = genID loc
renderLabel (StGlobalVar _) = undefined

{-|
  Code to take a list of pairs (pre/post) representing rules,
  and emit the corresponding stratego rewrite script as a
  single string.  This will be replaced at some point with code
  to take rules in StTree form, not linearized strings.
-}
strategoRules :: [(String,String)] -> String
strategoRules [] = error "Rule generation called with empty rule set."
strategoRules ts = header ++ rulesStr ++ footer
  where f = \(i,(t1,t2)) -> ("R" ++ (show i), " : " ++ t1 ++ " -> " ++ t2 ++ "\n")
        (labels,rules) = unzip $ map f (zip ([1..] :: [Integer]) ts)
        rulesStr = foldl1 (++) $ zipWith (\l r -> "  " ++ l ++ r) labels rules
        labelsStr = intercalate ";" $ map (\l -> "oncetd(" ++ l ++ ")") labels
        header = "module basic\n" ++
          "imports libstrategolib\n" ++
          "signature\n" ++
          "  sorts E F A\n" ++
          "constructors\n" ++
          "  gen_info               : F\n" ++
          "  file_info              : S * N * N -> F\n" ++
          "  add_op                 : E * E * A * F -> E\n" ++
          "  multiply_op            : E * E * A * F -> E\n" ++
          "  int_val                : A * F -> E\n" ++ 
          "  value_annotation       : N * F -> E\n" ++
          "  preprocessing_info     : S -> A\n" ++ 
          "  var_ref_exp            : A * F -> E\n" ++ 
          "  var_ref_exp_annotation : T * A * A * A * A -> A\n" ++
          "  binary_op_annotation   : T * A -> A\n" ++
          "  type_int               : T\n" ++
          "  default                : A\n" ++
          "  null                   : A\n" ++
          "rules\n" ++
          "  G : gen_info() -> file_info(\"compilerGenerated\",0,0)\n"
        footer = "strategies\n" ++
          "  main = io-wrap(rewr;gen)\n" ++
          "  rewr = " ++ labelsStr ++ "\n" ++
          "  gen = innermost(G)\n"

