#!/usr/bin/env bash

c2term -i input.c > input.trm
sed -i -e s/\'/\"/g input.trm
./basic -i input.trm > output.trm
sed -i -e s/\"/\'/g output.trm
term2c output.trm

