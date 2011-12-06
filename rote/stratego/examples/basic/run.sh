#!/usr/bin/env bash

c2term -i input.c > input.trm
sed -i -e s/\'/\"/g input.trm
pp-aterm -i input.trm -o input-pp.trm
./basic -i input.trm > output.trm
pp-aterm -i output.trm -o output-pp.trm
sed -i -e s/\"/\'/g output.trm
term2c output.trm

