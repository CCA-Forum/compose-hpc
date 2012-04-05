#!/usr/bin/env bash

sed -i -e s/\'/\"/g input.trm
./identity -i input.trm > output.trm
sed -i -e s/\"/\'/g output.trm
echo '.' >> output.trm
term2src output.trm

