#!/bin/sh

### set hostname (or username@hostname)
: ${MINITERMITE_HOST:?"Need MINITERMITE_HOST set."}

echo "Copying file."
scp $1 ${MINITERMITE_HOST}:sandbox/$1

echo "Running remote command."
ssh ${MINITERMITE_HOST} "/home/matt/src2term.sh $1" 

echo "Copying results back."
scp ${MINITERMITE_HOST}:sandbox/output.trm $2

echo "Fixing file to match aterm expectation."
cat $2 | sed s/\.$//g > $2.tmp
mv $2.tmp $2

echo "Cleaning up."
ssh ${MINITERMITE_HOST} "rm /home/matt/sandbox/output.trm"
ssh ${MINITERMITE_HOST} "rm /home/matt/sandbox/$1"
