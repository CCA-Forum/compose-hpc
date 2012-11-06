#!/bin/sh

### set hostname (or username@hostname)
: ${MINITERMITE_HOST:?"Need MINITERMITE_HOST set."}

if [ $# -lt 2 ]
then
  echo "Usage: $0 term_filename source_filename"
  exit
fi

echo "Copying file."
scp $1 ${MINITERMITE_HOST}:sandbox/$1

echo "Running remote command."
ssh ${MINITERMITE_HOST} "~/term2src.sh $1 $2" 

echo "Copying results back."
scp ${MINITERMITE_HOST}:sandbox/$2 $2

echo "Cleaning up."
ssh ${MINITERMITE_HOST} "rm sandbox/$2"
ssh ${MINITERMITE_HOST} "rm sandbox/$1"
