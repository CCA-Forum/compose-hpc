#!/usr/bin/env python
# -*- python -*-
## @package upc.makefile
#
# BRAID Upc Makefile generator
#
# Please report bugs to <adrian@llnl.gov>.
#
# \authors <pre>
#
# Copyright (c) 2011, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
#
# LLNL-CODE-473891.
# All rights reserved.
#
# This file is part of BRAID. For details, see
# http://compose-hpc.sourceforge.net/.
# Please read the COPYRIGHT file for Our Notice and
# for the BSD License.
#
# </pre>
#
from utils import write_to
import config

def gen_client(sidl_file, classes, prefix):
    """
    FIXME: make this a file copy from $prefix/share
    """
    files = prefix+'IORHDRS = '+' '.join([c+'_IOR.h' for c in classes])+'\n'
    files+= prefix+'STUBHDRS = '+' '.join(['{c}_Stub.h {c}_cStub.h'.format(c=c)
                                    for c in classes])+'\n'
    files+= prefix+'STUBSRCS = '+' '.join([c+'_Stub.upc' for c in classes])+'\n'
    write_to(prefix+'babel.make', files)

def gen_server(sidl_file, classes, pkgs, prefix):
    """
    FIXME: make this a file copy from $prefix/share
    """
    write_to(prefix+'babel.make', """
{prefix}IMPLHDRS =
{prefix}IMPLSRCS = {impls}
{prefix}IORHDRS = {iorhdrs} #FIXME Array_IOR.h
{prefix}IORSRCS = {iorsrcs}
{prefix}SKELSRCS = {skelsrcs}
{prefix}STUBHDRS = {stubhdrs}
{prefix}STUBSRCS = {stubsrcs}
""".format(prefix=prefix,
           impls=' '.join([p+'_Impl.upc'       for p in pkgs]),
           iorhdrs=' '.join([c+'_IOR.h'    for c in classes]),
           iorsrcs=' '.join([c+'_IOR.c'    for c in classes]),
           skelsrcs=' '.join([c+'_Skel.upc'  for c in classes]),
           stubsrcs=' '.join([c+'_Stub.upc' for c in classes]),
           stubhdrs=' '.join(['{c}_Stub.h {c}_cStub.h'.format(c=c)
                              for c in classes])))

def gen_gnumakefile(sidl_file):
    extraflags=''
    #extraflags='-ggdb -O0'
    write_to('GNUmakefile', r"""
# Generic Upc Babel wrapper GNU Makefile
#
# Copyright (c) 2008, 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the Components Team <components@llnl.gov>
# UCRL-CODE-2002-054
# All rights reserved.
#
# This file is part of Babel. For more information, see
# http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
# for Our Notice and the LICENSE file for the GNU Lesser General Public
# License.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
#
# You should have recieved a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# This Makefile uses GNU make extensions, so it may not work with
# other implementations of make. "

include babel.make
# please name the server library here
LIBNAME=impl
# please name the SIDL file here
SIDLFILE="""+sidl_file+r"""
# extra include/compile flags
EXTRAFLAGS="""+extraflags+r"""
# extra libraries that the implementation needs to link against
EXTRALIBS=
# library version number
VERSION=0.1.1
# PREFIX specifies the top of the installation directory
PREFIX=/usr/local
# the default installation installs the .la and .scl (if any) into the
# LIBDIR
LIBDIR=$(PREFIX)/lib
# the default installation installs the stub header and IOR header files
# in INCLDIR
INCLDIR=$(PREFIX)/include

UPC="""+config.UPC+r"""
UPCFLAGS="""+config.UPCFLAGS+r"""
UPC_VERSION="""+config.UPC_VERSION+r"""
BABEL_CC=$(shell babel-config --query-var=CC)
BABEL_INCLUDES=$(shell babel-config --includes) -I$(SIDL_RUNTIME)
BABEL_CFLAGS=$(shell babel-config --flags-c)
BABEL_LIBS=$(shell babel-config --libs-c-client)
UPC_CPPFLAGS="""+config.UPC_CPPFLAGS+r"""
UPC_CFLAGS="""+config.UPC_CFLAGS+r"""
UPC_LDFLAGS="""+config.UPC_LDFLAGS+r""" -L"""+config.PREFIX+r"""/lib
UPC_LIBS="""+config.UPC_LIBS+r""" -lsidlstub_upc -lupc_runtime_extra
SIDL_RUNTIME="""+config.PREFIX+r"""/include/upc

# most of the rest of the file should not require editing

ifeq ($(IMPLSRCS),)
  SCLFILE=
  BABELFLAG=--client=UPC
  MODFLAG=
else
  SCLFILE=lib$(LIBNAME).scl
  BABELFLAG=--server=UPC
  MODFLAG=-module
endif

all: lib$(LIBNAME).la $(SCLFILE) $(TARGET)

$(TARGET): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
	babel-libtool --mode=link $(BABEL_CC) -all-static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) $(EXTRA_LDFLAGS) -o $@

STUBOBJS=$(patsubst %.upc, %.lo, $(STUBSRCS:.c=.lo))
IOROBJS=$(IORSRCS:.upc=.lo)
SKELOBJS=$(SKELSRCS:.upc=.lo)
IMPLOBJS=$(IMPLSRCS:.upc=.lo)

PUREBABELGEN=$(IORHDRS) $(IORSRCS) $(STUBSRCS) $(STUBHDRS) $(SKELSRCS)
BABELGEN=$(IMPLHDRS) $(IMPLSRCS)

$(IMPLOBJS) : $(STUBHDRS) $(IORHDRS) $(IMPLHDRS)

lib$(LIBNAME).la : $(STUBOBJS) $(IOROBJS) $(IMPLOBJS) $(SKELOBJS)
	babel-libtool --mode=link --tag=CC $(UPC) $(UPCFLAGS) -o $@ \
	  -all-static \
          -release $(VERSION) \
	  -no-undefined $(MODFLAG) \
	  $(BABEL_CFLAGS) $(EXTRAFLAGS) $^ $(BABEL_LIBS) $(LIBS) \
	  $(UPC_LDFLAGS) $(UPC_LIBS) \
	  $(EXTRALIBS)

$(PUREBABELGEN) $(BABELGEN) : babel-stamp
# cf. http://www.gnu.org/software/automake/manual/automake.html#Multiple-Outputs
# Recover from the removal of $@
	@if test -f $@; then :; else \
	  trap 'rm -rf babel.lock babel-stamp' 1 2 13 15; \
true "mkdir is a portable test-and-set"; \
	  if mkdir babel.lock 2>/dev/null; then \
true "This code is being executed by the first process."; \
	    rm -f babel-stamp; \
	    $(MAKE) $(AM_MAKEFLAGS) babel-stamp; \
	    result=$$?; rm -rf babel.lock; exit $$result; \
	  else \
true "This code is being executed by the follower processes."; \
true "Wait until the first process is done."; \
	    while test -d babel.lock; do sleep 1; done; \
true "Succeed if and only if the first process succeeded." ; \
	    test -f babel-stamp; \
	  fi; \
	fi

babel-stamp: $(SIDLFILE)
	@rm -f babel-temp
	@touch babel-temp
	braid $(BABELFLAG) $(SIDLFILE)
	@mv -f babel-temp $@

lib$(LIBNAME).scl : $(IORSRCS)
ifeq ($(IORSRCS),)
	echo "lib$(LIBNAME).scl is not needed for client-side C bindings."
else
	-rm -f $@
	echo '<?xml version="1.0" ?>' > $@
	echo '<scl>' >> $@
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; fi ; \
	echo '  <library uri="'`pwd`/lib$(LIBNAME).la'" scope="'"$$scope"'" resolution="lazy" >' >> $@
	grep __set_epv $^ /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >>$@
	echo "  </library>" >>$@
	echo "</scl>" >>$@
endif

.SUFFIXES: .lo .upc

.c.lo:
	babel-libtool --mode=compile --tag=CC $(BABEL_CC) \
            $(BABEL_INCLUDES) $(BABEL_CFLAGS) $(EXTRAFLAGS) -c -o $@ $<

# $(UPC_CFLAGS) are automatically passed to cc by upcc
.upc.lo:
	babel-libtool --mode=compile --tag=UPC $(UPC) $(UPCFLAGS) -static \
	    $(BABEL_INCLUDES) $(EXTRAFLAGS) \
            -c -o $@ $<

clean :
	-rm -f $(PUREBABELGEN) babel-temp babel-stamp *.o *.lo

realclean : clean
	-rm -f lib$(LIBNAME).la lib$(LIBNAME).scl
	-rm -rf .libs

install : install-libs install-headers install-scl


install-libs : lib$(LIBNAME).la
	-mkdir -p $(LIBDIR)
	babel-libtool --mode=install install -c lib$(LIBNAME).la \
	  $(LIBDIR)/lib$(LIBNAME).la

install-scl : $(SCLFILE)
ifneq ($(IORSRCS),)
	-rm -f $(LIBDIR)/lib$(LIBNAME).scl
	-mkdir -p $(LIBDIR)
	echo '<?xml version="1.0" ?>' > $(LIBDIR)/lib$(LIBNAME).scl
	echo '<scl>' >> $(LIBDIR)/lib$(LIBNAME).scl
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; fi ; \
	echo '  <library uri="'$(LIBDIR)/lib$(LIBNAME).la'" scope="'"$$scope"'" resolution="lazy" >' >> $(LIBDIR)/lib$(LIBNAME).scl
	grep __set_epv $^ /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >>$(LIBDIR)/lib$(LIBNAME).scl
	echo "  </library>" >>$(LIBDIR)/lib$(LIBNAME).scl
	echo "</scl>" >>$(LIBDIR)/lib$(LIBNAME).scl
endif

install-headers : $(IORHDRS) $(STUBHDRS)
	-mkdir -p $(INCLDIR)
	for i in $^ ; do \
	  babel-libtool --mode=install cp $$i $(INCLDIR)/$$i ; \
	done

.PHONY: all clean realclean install install-libs install-headers install-scl
""")

