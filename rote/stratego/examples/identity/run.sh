#!/usr/bin/env bash

src2term -i input.c > input.trm
sed -i -e s/\'/\"/g input.trm
pp-aterm -i input.trm -o input-pp.trm
./identity -i input.trm > output.trm
pp-aterm -i output.trm -o output-pp.trm
sed -i -e s/\"/\'/g output.trm
echo '.' >>output.trm
term2src output.trm

