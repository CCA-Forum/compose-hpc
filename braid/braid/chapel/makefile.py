#!/usr/bin/env python
# -*- python -*-
## @package chapel.makefile
#
# BRAID Chapel Makefile generator
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

def generate_client(sidl_file, classes):
    """
    FIXME: make this a file copy from $prefix/share
    """
    files = 'IORHDRS = '+' '.join([c+'_IOR.h' for c in classes])+'\n'
    files+= 'STUBHDRS = '+' '.join(['{c}_Stub.h {c}_cStub.h'.format(c=c)
                                    for c in classes])+'\n'
    files+= 'STUBSRCS = '+' '.join([c+'_cStub.c' for c in classes])+'\n'
# this is handled by the use statement in the implementation instead:
# {file}_Stub.chpl
    write_to('babel.make', files)
    generate_client_server(sidl_file)


def generate_server(sidl_file, classes, pkgs):
    """
    FIXME: make this a file copy from $prefix/share
    """
    write_to('babel.make', """
IMPLHDRS =
IMPLSRCS = {impls}
IORHDRS = {iorhdrs} #FIXME Array_IOR.h
IORSRCS = {iorsrcs}
SKELSRCS = {skelsrcs}
STUBHDRS = {stubhdrs}
STUBSRCS = {stubsrcs}
""".format(impls=' '.join([p+'_Impl.chpl'       for p in pkgs]),
           iorhdrs=' '.join([c+'_IOR.h'    for c in classes]),
           iorsrcs=' '.join([c+'_IOR.c'    for c in classes]),
           skelsrcs=' '.join([c+'_Skel.c'  for c in classes]),
           stubsrcs=' '.join([c+'_cStub.c' for c in classes]),
           stubhdrs=' '.join([c+'_cStub.h' for c in classes])))
    generate_client_server(sidl_file)

def generate_client_server(sidl_file):
    extraflags=''
    #extraflags='-ggdb -O0'
    write_to('GNUmakefile', r"""
# Generic Chapel Babel wrapper GNU Makefile
#
# Copyright (c) 2008, 2012,Lawrence Livermore National Security, LLC.
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
# other implementations of make.

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

CHPL="""+config.CHPL+r"""
CHPL_MAKE_HOME="""+config.CHPL_ROOT+r"""
CHPL_MAKE_COMM="""+config.CHPL_COMM+r"""

CC=`babel-config --query-var=CC`
INCLUDES=`babel-config --includes` -I. -I$(CHPL_MAKE_HOME)/runtime/include -I$(SIDL_RUNTIME)
CFLAGS=`babel-config --flags-c` -std=c99
LIBS=`babel-config --libs-c-client`

CHPL_MAKE_MEM=default
CHPL_MAKE_COMPILER=gnu
CHPL_MAKE_TASKS=fifo
CHPL_MAKE_THREADS=pthreads

ifeq ($(CHPL_MAKE_COMM),gasnet)
CHPL_MAKE_SUBSTRATE_DIR=$(CHPL_MAKE_HOME)/lib/$(CHPL_HOST_PLATFORM)/$(CHPL_MAKE_COMPILER)/mem-$(CHPL_MAKE_MEM)/comm-gasnet-nodbg/substrate-udp/seg-none
else
CHPL_MAKE_SUBSTRATE_DIR=$(CHPL_MAKE_HOME)/lib/$(CHPL_HOST_PLATFORM)/$(CHPL_MAKE_COMPILER)/mem-$(CHPL_MAKE_MEM)/comm-none/substrate-none/seg-none
endif
####    include $(CHPL_MAKE_HOME)/runtime/etc/Makefile.include
CHPL=chpl --fast
# CHPL=chpl --print-commands --print-passes

include $(CHPL_MAKE_HOME)/make/Makefile.atomics

CHPL_FLAGS=-std=c99 \
  -DCHPL_TASKS_MODEL_H=\"tasks-fifo.h\" \
  -DCHPL_THREADS_MODEL_H=\"threads-pthreads.h\" \
  -I$(CHPL_MAKE_HOME)/runtime/include/tasks/$(CHPL_MAKE_TASKS) \
  -I$(CHPL_MAKE_HOME)/runtime/include/threads/$(CHPL_MAKE_THREADS) \
  -I$(CHPL_MAKE_HOME)/runtime/include/comm/none \
  -I$(CHPL_MAKE_HOME)/runtime/include/comp-gnu \
  -I$(CHPL_MAKE_HOME)/runtime/include/$(CHPL_HOST_PLATFORM) \
  -I$(CHPL_MAKE_HOME)/runtime/include/atomics/$(CHPL_MAKE_ATOMICS) \
  -I$(CHPL_MAKE_HOME)/runtime/include/mem/$(CHPL_MAKE_MEM) \
  -I$(CHPL_MAKE_HOME)/runtime/include \
  -I. -Wno-all 

CHPL_LDFLAGS=-L$(CHPL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHPL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm  -lpthread -lsidlstub_chpl

CHPL_GASNET_LDFLAGS=-L$(CHPL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHPL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm -lpthread -L$(CHPL_MAKE_HOME)/third-party/gasnet/install/$(CHPL_HOST_PLATFORM)-$(CHPL_MAKE_COMPILER)/seg-everything/nodbg/lib -lgasnet-udp-par -lamudp -lpthread -lgcc -lm

CHPL_LAUNCHER_LDFLAGS=$(CHPL_MAKE_SUBSTRATE_DIR)/launch-amudprun/main_launcher.o
LAUNCHER_LDFLAGS=-L$(CHPL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads -L$(CHPL_MAKE_SUBSTRATE_DIR)/launch-amudprun -lchpllaunch -lchpl -lm

SIDL_RUNTIME="""+config.PREFIX+r"""/include/runtime
CHPL_HEADERS=-I$(SIDL_RUNTIME) -M$(SIDL_RUNTIME) \
  chpl_sidl_array.h $(SIDL_RUNTIME)/sidl_*_Stub.h

# most of the rest of the file should not require editing

ifeq ($(IMPLSRCS),)
  SCLFILE=
  BABELFLAG=--client=Chapel
  MODFLAG=
else
  SCLFILE=lib$(LIBNAME).scl
  BABELFLAG=--server=Chapel
  MODFLAG=-module
  DCE= #--no-dead-code-elimination # include everything in libimpl.la
endif

ifeq ($(CHPL_MAKE_COMM),gasnet)

all: lib$(LIBNAME).la $(SCLFILE) $(TARGET) $(TARGET)_real

# actual program
$(TARGET)_real: lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
	babel-libtool --mode=link $(CXX) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) \
          $(CHPL_GASNET_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

# launcher
$(TARGET): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo
	echo "#include \"chplcgfns.h\"" > $(IMPL).chpl.dir/config.c
	echo "#include \"config.h\""   >> $(IMPL).chpl.dir/config.c
	echo "#include \"_config.c\""  >> $(IMPL).chpl.dir/config.c
	babel-libtool --mode=compile --tag=CC $(CC) \
          -std=c99 -I$(CHPL_MAKE_HOME)/runtime/include/$(CHPL_HOST_PLATFORM) \
	  -I$(CHPL_MAKE_HOME)/runtime/include -I. \
	  $(IMPL).chpl.dir/config.c -c -o $@.lo
	babel-libtool --mode=link $(CC) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $@.lo $(SERVER) \
          $(CHPL_LAUNCHER_LDFLAGS) $(LAUNCHER_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

else

all: lib$(LIBNAME).la $(SCLFILE) $(TARGET)

$(TARGET): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
	babel-libtool --mode=link $(CC) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) $(CHPL_LDFLAGS) $(EXTRA_LDFLAGS) -o $@
endif

STUBOBJS=$(patsubst .chpl, .lo, $(STUBSRCS:.c=.lo))
IOROBJS=$(IORSRCS:.c=.lo)
SKELOBJS=$(SKELSRCS:.c=.lo)
IMPLOBJS=$(IMPLSRCS:.chpl=.lo)

PUREBABELGEN=$(IORHDRS) $(IORSRCS) $(STUBSRCS) $(STUBHDRS) $(SKELSRCS)
BABELGEN=$(IMPLHDRS) $(IMPLSRCS)

$(IMPLOBJS) : $(STUBHDRS) $(IORHDRS) $(IMPLHDRS)

lib$(LIBNAME).la : $(STUBOBJS) $(IOROBJS) $(IMPLOBJS) $(SKELOBJS)
	babel-libtool --mode=link --tag=CC $(CC) -o lib$(LIBNAME).la \
	  -static \
          -release $(VERSION) \
	  -no-undefined $(MODFLAG) \
	  $(CFLAGS) $(EXTRAFLAGS) $^ $(LIBS) \
          $(CHPL_LDFLAGS) -lchpl \
	  $(EXTRALIBS)
 #-rpath $(LIBDIR) 

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

.SUFFIXES: .lo .chpl

.c.lo:
	babel-libtool --mode=compile --tag=CC $(CC) $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) -c -o $@ $<

ifeq ($(IMPLSRCS),)
.chpl.lo:
	$(CHPL) --savec $<.dir $< $(IORHDRS) $(STUBHDRS) $(CHPL_HEADERS) $(DCE) --make true  # gen C-code only
	babel-libtool --mode=compile --tag=CC $(CC) \
            -I./$<.dir $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) \
            $(CHPL_FLAGS) -c -o $@ $<.dir/_main.c
else
.chpl.lo:
	$(CHPL) --library --savec $<.dir $< $(IORHDRS) $(STUBHDRS) $(CHPL_HEADERS) $(DCE) --make true  # gen C-code
	#headerize $<.dir/_config.c $<.dir/Chapel*.c $<.dir/Default*.c $<.dir/DSIUtil.c $<.dir/chpl*.c $<.dir/List.c $<.dir/Math.c $<.dir/Search.c $<.dir/Sort.c $<.dir/Types.c
	#perl -pi -e 's/((chpl__autoDestroyGlobals)|(chpl_user_main)|(chpl__init)|(chpl_main))/$*_\1/g' $<.dir/$*.c
	perl -pi -e 's|^  if .$*|  chpl_bool $*_chpl__init_$*_p = false;\n  if ($*|' $<.dir/$*.c
	echo '#include "../_chplmain.c"' >>$<.dir/_main.c
	babel-libtool --mode=compile --tag=CC $(CC) \
            -I./$<.dir $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) \
            $(CHPL_FLAGS) -c -o $@ $<.dir/_main.c
endif


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

