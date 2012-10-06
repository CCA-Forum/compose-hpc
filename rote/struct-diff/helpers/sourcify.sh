#!/bin/sh

echo "Fixing quotes."
cat $1 | sed s/\"/\'/g > $1.tmp

echo "Copying file to ftt."
scp $1.tmp matt@ftt.dev.galois.com:sandbox/$1
rm $1.tmp

echo "Running remote command."
ssh matt@ftt.dev.galois.com "/home/matt/term2src.sh $1 $2" 

echo "Copying results back."
scp matt@ftt.dev.galois.com:sandbox/$2 $2

echo "Cleaning up."
ssh matt@ftt.dev.galois.com "rm /home/matt/sandbox/$2"
ssh matt@ftt.dev.galois.com "rm /home/matt/sandbox/$1"
