#!/bin/sh

### set hostname (or username@hostname)
: ${MINITERMITE_HOST:?"Need MINITERMITE_HOST set."}

echo "Copying file."
scp $1 ${MINITERMITE_HOST}:sandbox/$1

echo "Running remote command."
ssh ${MINITERMITE_HOST} "/home/matt/term2src.sh $1 $2" 

echo "Copying results back."
scp ${MINITERMITE_HOST}:sandbox/$2 $2

echo "Cleaning up."
ssh ${MINITERMITE_HOST} "rm /home/matt/sandbox/$2"
ssh ${MINITERMITE_HOST} "rm /home/matt/sandbox/$1"
