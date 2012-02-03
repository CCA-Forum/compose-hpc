#!/bin/bash

# setup for tux314

export SH=bash
export PACKAGE=braid
export SNAPSHOT_NUMBER=`date '+%Y%m%d'`
export PREFIX=$HOME/work/babel/install
export GANTLETDIR=$PREFIX/../babel/regression
export PATH=$HOME/sw_ubgl/bin:$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:/usr/lib/jvm/java-6-openjdk/jre/lib/amd64/server:$HOME/sw/lib64:$HOME/sw/lib
host=`hostname`
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig
export GIT=git
export COMPOSE_REPO=ssh://adrian_prantl@compose-hpc.git.sourceforge.net/gitroot/compose-hpc/compose-hpc
export URL=http://www.cca-forum.org/download/babel
export MAKE=make
JOBS=`cat /proc/cpuinfo |sed 's/ //g' |awk 'BEGIN {FS=":"; n=0} {if ($1 ~ "processor") n=$2} END {print n+1}'`
export MAKE_FLAGS="-j $JOBS -l $JOBS"
export MAIL=mail
export PERL=/usr/bin/perl
export CD=cd
export MKDIR=mkdir
export MV=mv
export CHMOD=chmod
export CHGRP=chgrp
export TESTGID=babel
export FIND=find
export PACKAGING_BUILDDIR=/tmp/`whoami`/braid_scratch
mkdir -p ${PACKAGING_BUILDDIR}
export LDFLAGS="-L$HOME/sw/lib64"
export PYTHONPATH=$PREDIX/lib/python2.6/site-packages
export CONFIG_FLAGS="" 
export PROFILE_NAME="casc-ubuntu-10.4"
export MAIL_SERVER=localhost
