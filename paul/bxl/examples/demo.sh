#!/bin/bash

BXL="python ../bxl.py"
CONTRACT="/tmp/_contract"
ATERM="/tmp/_aterm"

$BXL --version

cat >$CONTRACT <<EOF
REQUIRE
  pos_weights: ((weights!=NULL) and (len>0)) implies pce_all(weights>((0+1)*0), len);
EOF
echo "Converting the following annotation to aterms:"
cat $CONTRACT
echo
echo "> $BXL -f contract2aterm.bxl $CONTRACT $ATERM"
        $BXL -f contract2aterm.bxl $CONTRACT $ATERM || exit 1
cat $ATERM
echo
echo "Converting the aterms back into an annotation"
echo "> $BXL -f aterm2contract.bxl $ATERM"
        $BXL -f aterm2contract.bxl $ATERM || exit 2
echo
rm $CONTRACT $ATERM

