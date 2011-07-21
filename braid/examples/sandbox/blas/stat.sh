#!/bin/bash

function pull_info() {
    for f in $@ #`ls -1 data_hybrid*`; do
      do
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

function median() {
    python -c "
import numpy,sys
for l in sys.stdin.readlines():
    print ' '.join(l.split()[:4]+[str(numpy.mean(
        map(lambda x: float(x), l.split()[5:])))])
"
}

echo "Name Locales Elements Blocksize Time"
pull_info /p/lscratchb/prantl1/data_pure* | merge_runs | median |sort -s -n -k 5 |sort -s -n -k 3 |sort -s -n -k 2 |sort -s -n -k 4 