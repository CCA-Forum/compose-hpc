#!/usr/bin/env bash

src2term --stratego -i input.c -o input.trm
./basic -i input.trm > output.trm
term2src --stratego output.trm

