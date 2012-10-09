#!/bin/sh

echo "Copying file to ftt."
scp $1 matt@ftt.dev.galois.com:sandbox/$1

echo "Running remote command."
ssh matt@ftt.dev.galois.com "/home/matt/src2term.sh $1" 

echo "Copying results back."
scp matt@ftt.dev.galois.com:sandbox/output.trm $2

echo "Fixing quotes and trailing dot."
cat $2 | sed s/\'/\"/g > $2.tmp
mv $2.tmp $2
cat $2 | sed s/\.$//g > $2.tmp
mv $2.tmp $2


echo "Cleaning up."
ssh matt@ftt.dev.galois.com "rm /home/matt/sandbox/output.trm"
ssh matt@ftt.dev.galois.com "rm /home/matt/sandbox/$1"
