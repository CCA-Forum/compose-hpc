#!/bin/sh

source ${HOME}/compose_rose_setup.sh

export PATH=$PATH:$ROSEINSTALL/bin

src2term --stratego ${HOME}/sandbox/$1 -o ${HOME}/sandbox/output.trm
