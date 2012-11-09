#!/bin/sh

source ${HOME}/compose_rose_setup.sh

export PATH=$PATH:$ROSEINSTALL/bin

term2src --stratego ${HOME}/sandbox/$1 -o ${HOME}/sandbox/$2
