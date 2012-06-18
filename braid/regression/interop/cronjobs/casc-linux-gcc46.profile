#!/bin/sh
# 
# Profile: casc-linux-gcc46.profile
#
export MAIL=/usr/bin/mail
export MAIL_SERVER=poptop.llnl.gov
export SH=bash
export PACKAGE=babel
export SNAPSHOT_NUMBER=`date '+%Y%m%d'`
export PREFIX=$HOME/babel/install
export PATH=$PREFIX/bin:$HOME/sw/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$HOME/sw/lib
export LDFLAGS=-L/home/prantl1/sw/lib64 
export PYTHON=/usr/bin/python
host=`hostname`
export SVN=svn
export SVNUSER=
export SVNROOT=svn+ssh://${SVNUSER}svn.cca-forum.org/svn/babel/trunk
export URL=http://www.cca-forum.org/download/babel
export MAKE=make
JOBS=`sysctl hw.ncpu | awk '{print $2}'`
export MAKE_FLAGS="-j $JOBS -l $JOBS"
export MAIL=mail
export PERL=perl
export CD=cd
export MKDIR=mkdir
export MV=mv
export CHMOD=chmod
export CHGRP=chgrp
export TESTGID=babel
export FIND=find
export PACKAGING_BUILDDIR=/tmp/babel_scratch
mkdir -p ${PACKAGING_BUILDDIR}
export CPP='gcc-fsf-4.6 -E'
export CC=gcc-fsf-4.6
export CXX=g++-fsf-4.6
export FC=gfortran-fsf-4.6
export F77=gfortran-fsf-4.6
FLAGS='-O2 -ggdb -pipe -march=native -funroll-loops -ftree-vectorize'
export CPP='gcc -E'
export CFLAGS="$FLAGS"
export CXXFLAGS="$FLAGS"
export FFLAGS="$FLAGS"
export FCFLAGS="$FLAGS"
export CONFIG_FLAGS="" 
export PROFILE_NAME="casc-linux.profile"
