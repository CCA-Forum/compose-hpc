#!sh
checkTrace() {
  echo
  echo "Comparing: tce-$1-$2-trace.csv tce-$1-Always-trace.csv.."
  #sdiff -s tce-$1-$2-trace.csv tce-$1-Always-trace.csv
  sdiff tce-$1-$2-trace.csv tce-$1-Always-trace.csv
}

clauses="Asrt Inv InvAsrt InvPost InvPre InvPreAsrt InvPrePost InvPrePostAsrt None Post Pre PreAsrt PrePost PrePostAsrt"
freq="AdaptiveFit AdaptiveTiming Periodic Random"

for cl in $clauses; do
  for fr in $freq; do
     checkTrace $cl $fr
  done
done

echo; echo "DONE"
