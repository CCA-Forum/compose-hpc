
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
LIBNAME=pgas
# please name the SIDL file here
SIDLFILE= ../pgas_array.sidl
# extra include/compile flags
EXTRAFLAGS=
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

CHPL_MAKE_HOME=/g/g91/prantl1/sw/opt/chapel

# use Chapel runtime's CC 
BABEL_CC=$(CC)
# BABEL_CC=$(shell babel-config --query-var=CC)
BABEL_INCLUDES=$(shell babel-config --includes) -I. -I$(CHPL_MAKE_HOME)/runtime/include -I$(CHPL_MAKE_HOME)/third-party/utf8-decoder -I$(SIDL_RUNTIME)
BABEL_CFLAGS=$(patsubst -Wall,, $(shell babel-config --flags-c))
BABEL_LIBS=$(shell babel-config --libs-c-client)

# triggers -DCHPL_OPTIMIZE
OPTIMIZE=1

# Get runtime headers and required -D flags.
include $(CHPL_MAKE_HOME)/runtime/etc/Makefile.include

COMP_GEN_CFLAGS=$(WARN_GEN_CFLAGS) $(IEEE_FLOAT_GEN_CFLAGS)
CHPL_CFLAGS_FULL=$(GEN_CFLAGS) $(COMP_GEN_CFLAGS) $(CHPL_RT_INC_DIR)
CHPL_FLAGS=$(patsubst -Wmissing-declarations,, \
	   $(patsubst -W%-prototypes,, \
           $(patsubst -Werror,, \
           $(CHPL_CFLAGS_FULL)))) \
	   $(CHPL_GASNET_CFLAGS) \
	   $(GASNET_OPT_CFLAGS)

CHPL_LDFLAGS= \
 $(GEN_LFLAGS) $(COMP_GEN_LFLAGS) \
 -L$(CHPL_RT_LIB_DIR) \
 -lchpl -lm -lpthread -lsidlstub_chpl -lsidl

CHPL_GASNET_LDFLAGS= \
  $(GEN_LFLAGS) $(COMP_GEN_LFLAGS) \
  -L$(CHPL_RT_LIB_DIR) \
  -lchpl -lm -lpthread \
  $(CHPL_GASNET_LFLAGS)

CHPL_LAUNCHER_LDFLAGS=$(CHPL_MAKE_SUBSTRATE_DIR)/launch-amudprun/main_launcher.o
LAUNCHER_LDFLAGS=-L$(CHPL_MAKE_SUBSTRATE_DIR)/tasks-$(CHPL_MAKE_TASKS)/threads-$(CHPL_MAKE_THREADS) -L$(CHPL_MAKE_SUBSTRATE_DIR)/launch-amudprun -lchpllaunch -lchpl -lm

SIDL_RUNTIME=/g/g91/prantl1/install/include/chpl
CHPL_HEADERS=-I$(SIDL_RUNTIME) -M$(SIDL_RUNTIME) \
  $(SIDL_RUNTIME)/sidl_BaseClass_IOR.h chpl_sidl_array.h $(SIDL_RUNTIME)/sidl_*_Stub.h

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

all: lib$(LIBNAME).la $(SCLFILE) $(TARGET)

# actual program
$(TARGET)_real: lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
	babel-libtool --mode=link $(CXX) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) \
          $(CONDUIT_LIBS) $(CHPL_GASNET_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

# launcher
$(TARGET): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo $(TARGET)_real
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
	babel-libtool --mode=link $(BABEL_CC) -static lib$(LIBNAME).la \
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
	babel-libtool --mode=link --tag=CC $(BABEL_CC) -o lib$(LIBNAME).la \
          -release $(VERSION) \
	  -no-undefined $(MODFLAG) \
	  $(BABEL_CFLAGS) $(EXTRAFLAGS) $^ $(BABEL_LIBS) $(LIBS) \
          $(CHPL_LDFLAGS) \
          $(CHPL_GASNET_LFLAGS) \
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
	babel-libtool --mode=compile --tag=CC $(BABEL_CC) $(CHPL_FLAGS) $(BABEL_INCLUDES) $(BABEL_CFLAGS) $(EXTRAFLAGS) -c -o $@ $<


# Chapel options used:
#
# --savec [dir]  save the generated C code in dir
# --make [cmd]   used to disable compilation of C code by chpl by using "true" as $MAKE
# --devel        turn on more verbose error output
# --library      compile a library
# --fast         optimize
# --print-commands --print-passes
#
# NOTE
# ----
# Because of the order of the -I options, it is crucial that
# CHPL_FLAGS come before BABEL_CFLAGS and friends.
#
ifeq ($(IMPLSRCS),)
.chpl.lo:
	$(CHPL) --fast --devel --savec $<.dir  --make true $< \
            $(STUBHDRS) $(CHPL_HEADERS) $(DCE)
	babel-libtool --mode=compile --tag=CC $(BABEL_CC) \
            -I./$<.dir \
	    $(CHPL_FLAGS) \
	    $(BABEL_INCLUDES) $(BABEL_CFLAGS) $(EXTRAFLAGS) \
            -c -o $@ $<.dir/_main.c
else
.chpl.lo:
	@echo +++++++
	$(CHPL) --fast --devel --library --savec $<.dir --make true $< \
	    $(STUBHDRS) $(CHPL_HEADERS) $(DCE)
	@echo +++++++
	babel-libtool --mode=compile --tag=CC $(BABEL_CC) \
            -I./$<.dir \
	    $(CHPL_FLAGS) \
            $(BABEL_INCLUDES) $(BABEL_CFLAGS) $(EXTRAFLAGS) \
            -c -o $@ $<.dir/_main.c
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
