#!/usr/bin/env bash

src2term --stratego -i input.c -o input.trm
pp-aterm -i input.trm -o input-pp.trm
./identity -i input.trm > output.trm
pp-aterm -i output.trm -o output-pp.trm
term2src --stratego output.trm

