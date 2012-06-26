#!/bin/sh

echo "This test checks whether the IOR generated by BRAID and Babel is
compatible by comparing the <sizeof()> of all the structs."

if [ $# -lt 2 ]; then 
    echo "usage: $0 libC runChapel -I[prefix]/include ..."
    exit 1
fi


babel=$1
braid=$2
shift; shift

for babel_ior in $babel/*IOR.h; do
  file=`basename $babel_ior`
  braid_ior="$braid/$file"
  printf "Comparing $babel_ior and $braid_ior... "
  if [ ! -e $braid_ior ]; then
      echo "SKIP (braid version does not exist)"
      continue
  fi

  echo "#include \"$babel_ior\"" >babel.c
  echo "#include \"$braid_ior\"" >braid.c
  echo "#include \"common.c\""   >>babel.c
  echo "#include \"common.c\""   >>braid.c
  echo "#include <stdio.h>"      >common.c
  echo "int main() {"            >>common.c
  awk -- '
    /^struct .* \{/ { print "  printf(\"sizeof("$1" "$2") = %ld\\n\", sizeof("$1" "$2"));" }
    /.*Anonymous class definition.*/ {exit}
  ' $babel_ior                  >>common.c
  echo "return 0; }"            >>common.c

  (      gcc $@ -I$babel -o babel babel.c \
      && gcc $@ -I$braid -o braid braid.c )
  if [ $? -ne 0 ]; then
      echo "** COMPILE ERROR *******************"
      continue 
  fi
  ./babel >babel.out 
  ./braid >braid.out 
  diff -q babel.out braid.out
  if [ $? -ne 0 ]; then 
      echo
      diff babel.out braid.out
      echo "** FAIL ***************"
  else
      echo "OK!"
  fi
done

