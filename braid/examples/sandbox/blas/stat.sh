#!/bin/bash

# Usage: $0 file*.txt

# collate all the info from the bench*.txt files in $@
# into a space-seperated table
function pull_info() {
    for f in $@; do
      if grep -q SUCCESS $f; then
	  true
      else
	  echo "**ERROR: test $f *FAILED*!" 1>&2
	  continue
      fi

      time=`grep "^Execution time" $f | awk '{print $4}'`
      f=`echo $f|sed -e 's/\\.txt\$//'`

         name=`echo $f |awk -F - '{ print $1; }'`
      locales=`echo $f |awk -F - '{ print $2; }'`
        elems=`echo $f |awk -F - '{ print $3; }'`
      blkSize=`echo $f |awk -F - '{ print $4; }'`
        runId=`echo $f |awk -F - '{ print $5; }'`
      printf "$name $locales $elems $blkSize $runId $time\n"
    done
}

# bench1  a b c 1 time1
# bench1  a b c 2 time2
# bench2  d e f 1 time3
# ----->
# bench1  a b c time1 time2
# bench2  d e f 3
function merge_runs() {
    awk '{
      bench = $1 " " $2 " " $3 " " $4;
      time = $6;
      bs[bench] = bs[bench] " " $6 " ";
    }
    END {
	for (b in bs) {
            print b " " bs[b]
        }
    }'
}

# bench1  a b c time1 time2
# -------->
# bench1  a b c `median(time1 time2)`
function median() {
    python -c "
import numpy,sys
for l in sys.stdin.readlines():
    print ' '.join(l.split()[:4]+[str(numpy.median(
        map(lambda x: float(x), l.split()[4:])))])
"
}

echo "Name Locales Elements Blocksize Time"
pull_info $@ | merge_runs | median |sort -s -n -k 5 |sort -s -n -k 3 |sort -s -n -k 2 |sort -s -n -k 4 