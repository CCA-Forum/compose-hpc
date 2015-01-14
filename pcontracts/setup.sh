#!/bin/sh

addToLoadLib() {
    LIB=$1
    if [ $LD_LIBRARY_PATH ]; then
      echo $LD_LIBRARY_PATH | grep -q ${LIB}
      if [ $? -ne 0 ]; then
        export LD_LIBRARY_PATH="$LIB:$LD_LIBRARY_PATH"
      fi
    else
      export LD_LIBRARY_PATH="$LIB"
    fi
}


# FIXME: BOOST_HOME and ROSE_HOME
export BOOST_HOME=/change/me/to/path/to/boost/install/directory
export ROSE_HOME=/change/me/to/path/to/rose/install/direcotry

export PCONTRACTS_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; pwd )"

# FIXME: libjvm.so path
addToLoadLib /change/me/to/path/to/java/install/libjvm.so/directory

addToLoadLib $ROSE_HOME/lib
