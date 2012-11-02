#!/bin/sh

### set hostname (or username@hostname)
: ${MINITERMITE_HOST:?"Need MINITERMITE_HOST set."}

if [ $# -lt 2 ]
then
  echo "Usage: $0 source_filename term_filename"
  exit
fi

echo "Copying file."
scp $1 ${MINITERMITE_HOST}:sandbox/$1

echo "Running remote command."
ssh ${MINITERMITE_HOST} "~/src2term.sh $1" 

echo "Copying results back."
scp ${MINITERMITE_HOST}:sandbox/output.trm $2

echo "Fixing file to match aterm expectation."
cat $2 | sed s/\.$//g > $2.tmp
mv $2.tmp $2

echo "Cleaning up."
ssh ${MINITERMITE_HOST} "rm sandbox/output.trm"
ssh ${MINITERMITE_HOST} "rm sandbox/$1"
