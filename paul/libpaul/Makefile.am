## Process this file with automake to create Makefile.in.

ACLOCAL_AMFLAGS = -I m4

SUBDIRS=src tests 
DIST_SUBDIRS = 
DISTCHECK_CONFIGURE_FLAGS=--with-rosedir=$(ROSEDIR) --with-boostdir=$(BOOSTDIR) --with-boost-compiler-string=$(BOOST_COMPILER_STRING) CXX=$(CXX)

.PHONY: libpaul
