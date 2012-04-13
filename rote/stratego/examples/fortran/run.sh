#!/usr/bin/env bash

src2term --stratego -Iinput -i input/test.F -o input.trm
./identity -i input.trm > output.term
term2src --stratego output.term

