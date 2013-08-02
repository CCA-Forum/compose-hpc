#!sh
showHeader() {
  #res=`head -n 1 $1 | sed "s/^[^;]*;[^;]*;\([^;]*;\)[^;]*; /\1 /g"`
  res=`head -n 1 $1 | sed "s/^[^;]*;[^;]*;[^;]*;[^;]*; //g"`
  echo $res
}

trimContents() {
  #cat $1 | sed "s/^[^;]*;[^;]*;\([^;]*;\)[^;]*; /\1 /g" > $2
  cat $1 | sed "s/^[^;]*;[^;]*;[^;]*;[^;]*; //g" > $2
}

checkStats() {
  echo; echo
  echo "Comparing: tce-$1-$2-stats.csv tce-$1-Always-stats.csv.."
  trimContents tce-$1-$2-stats.csv $1-$2.tmp
  trimContents tce-$1-Always-stats.csv $1-Always.tmp
  diff $1-$2.tmp $1-Always.tmp > $1.diffs
  if [ -s $1.diffs ]; then
    echo
    showHeader tce-$1-Always-stats.csv
    cat $1.diffs
  fi
  rm -f $1-$2.tmp $1-Always.tmp $1.diffs 
}

clauses="Asrt Inv InvAsrt InvPost InvPre InvPreAsrt InvPrePost InvPrePostAsrt None Post Pre PreAsrt PrePost PrePostAsrt"
freq="AdaptiveFit AdaptiveTiming Periodic Random"

for cl in $clauses; do
  for fr in $freq; do
     checkStats $cl $fr
  done
done

echo; echo "DONE"
