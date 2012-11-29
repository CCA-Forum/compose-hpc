#!/bin/bash

python bxl.py --version

cat >_contract <<EOF
REQUIRE
  pos_weights: ((weights!=NULL) and (len>0)) implies pce_all(weights>((0+1)*0), len);
EOF
echo "Converting the following annotation to aterms:"
cat _contract
echo
echo "> python bxl.py -f contract2aterm.bxl _contract _aterm"
        python bxl.py -f contract2aterm.bxl _contract _aterm 
cat _aterm
echo
echo "Converting the aterms back into an annotation"
echo "> python bxl.py -f aterm2contract.bxl _aterm"
        python bxl.py -f aterm2contract.bxl _aterm
echo
rm _contract _aterm

