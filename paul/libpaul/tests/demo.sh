#!/bin/bash

python bxl.py --version

cat >_contract <<EOF
REQUIRE
  pos_weights: ((weights!=NULL) and (len>0)) implies pce_all(weights>((0+1)*0), len);
EOF
echo "Converting the following annotation to sexps:"
cat _contract
echo
echo "> ./contract2sexp.bxl _contract _sexp"
        ./contract2sexp.bxl _contract _sexp 
cat _sexp
echo
echo "Converting the sexps back into an annotation"
echo "> ./sexp2contract.bxl _sexp"
        ./sexp2contract.bxl _sexp
echo
rm _contract _sexp

