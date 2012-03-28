#!/usr/bin/env bash

src2term -i input.c > input.trm
sed -i -e s/\'/\"/g -e 's/, ::/, "::"/g' input.trm
./basic -i input.trm > output.trm
sed -i -e s/\"/\'/g -e 's/, "::"/, ::/g' output.trm
echo '.' >>output.trm
term2src output.trm

