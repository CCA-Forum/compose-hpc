
# Generic Chapel Babel wrapper GNU Makefile
# $Id$
#
# Copyright (c) 2008, Lawrence Livermore National Security, LLC.
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
SIDLFILE=../blas.sidl
# extra include/compile flags
EXTRAFLAGS=-ggdb -O0 
# EXTRAFLAGS=-ggdb -O0 -I/usr/include/atlas

# extra libraries that the implementation needs to link against
# EXTRALIBS=-lcblas  
EXTRALIBS=-lcblas -L/usr/lib64/atlas 

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
CHAPEL=$(CHPL_HOME)/bin/$(CHPL_HOST_PLATFORM)/chpl
CHAPEL_ROOT=$(CHPL_HOME)
CHAPEL_MAKE_MEM=default
# CHAPEL_MAKE_COMM=none
CHAPEL_MAKE_COMM=gasnet
CHAPEL_MAKE_COMPILER=gnu
CHAPEL_MAKE_TASKS=fifo
CHAPEL_MAKE_THREADS=pthreads

ifeq ($(CHAPEL_MAKE_COMM),gasnet)
#CHAPEL_MAKE_SUBSTRATE_DIR=$(CHAPEL_ROOT)/lib/$(CHPL_HOST_PLATFORM)/$(CHAPEL_MAKE_COMPILER)/comm-gasnet-nodbg/substrate-udp
CHAPEL_MAKE_SUBSTRATE_DIR=$(CHAPEL_ROOT)/lib/$(CHPL_HOST_PLATFORM)/$(CHAPEL_MAKE_COMPILER)/comm-gasnet-nodbg/substrate-ibv
else
CHAPEL_MAKE_SUBSTRATE_DIR=$(CHAPEL_ROOT)/lib/$(CHPL_HOST_PLATFORM)/$(CHAPEL_MAKE_COMPILER)/comm-none/substrate-none
endif
####    include $(CHAPEL_ROOT)/runtime/etc/Makefile.include
CHPL=chpl
# CHPL=chpl --print-commands --print-passes

CHPL_FLAGS=-std=c99 -DGASNET_PAR -D_REENTRANT -D_GNU_SOURCE -l$(CHAPEL_MAKE_COMPILER)/seg-everything/nodbg/include -l$(CHAPEL_MAKE_COMPILER)/seg-everything/nodbg/include/ibv-conduit -DGASNET_CONDUIT_IBV -DCHPL_TASKS_H=\"tasks-fifo.h\" -DCHPL_THREADS_H=\"threads-pthreads.h\" -I$(CHAPEL_ROOT)/runtime/include/tasks/fifo -I$(CHAPEL_ROOT)/runtime/include/threads/pthreads -I$(CHAPEL_ROOT)/runtime/include/comm/gasnet -I$(CHAPEL_ROOT)/runtime/include/comp-gnu -I$(CHAPEL_ROOT)/runtime/include/$(CHPL_HOST_PLATFORM) -I$(CHAPEL_ROOT)/runtime/include -I. -Wno-all
#CHPL_FLAGS=-std=c99 -DCHPL_TASKS_H=\"tasks-fifo.h\" -DCHPL_THREADS_H=\"threads-pthreads.h\" -I$(CHAPEL_ROOT)/runtime/include/tasks/fifo -I$(CHAPEL_ROOT)/runtime/include/threads/pthreads -I$(CHAPEL_ROOT)/runtime/include/comm/gasnet -I$(CHAPEL_ROOT)/runtime/include/comp-gnu -I$(CHAPEL_ROOT)/runtime/include/$(CHPL_HOST_PLATFORM) -I$(CHAPEL_ROOT)/runtime/include -I. -Wno-all 

CHPL_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm  -lpthread

#CHPL_GASNET_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm -lpthread -L$(CHAPEL_ROOT)/third-party/gasnet/install/$(CHPL_HOST_PLATFORM)-$(CHAPEL_MAKE_COMPILER)/seg-everything/nodbg/lib -lgasnet-udp-par -lamudp -lpthread -lgcc -lm
CHPL_GASNET_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm -lpthread -L$(CHAPEL_ROOT)/third-party/gasnet/install/$(CHPL_HOST_PLATFORM)-$(CHAPEL_MAKE_COMPILER)/seg-everything/nodbg/lib -lgasnet-ibv-par -libverbs -lpthread -lgcc -lm

#CHPL_LAUNCHER_LDFLAGS=$(CHAPEL_MAKE_SUBSTRATE_DIR)/launch-amudprun/main_launcher.o
CHPL_LAUNCHER_LDFLAGS=$(CHAPEL_MAKE_SUBSTRATE_DIR)/launch-gasnetrun_ibv/main_launcher.o

#LAUNCHER_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads -L$(CHAPEL_MAKE_SUBSTRATE_DIR)/launch-amudprun -lchpllaunch -lchpl -lm
LAUNCHER_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads -L$(CHAPEL_MAKE_SUBSTRATE_DIR)/launch-gasnetrun_ibv -lchpllaunch -lchpl -lm

#SIDL_RUNTIME=/Users/imam1/softwares/include
SIDL_RUNTIME=/g/g91/prantl1/install/include

CHPL_HEADERS=-I$(SIDL_RUNTIME)/chpl -M$(SIDL_RUNTIME)/chpl \
  chpl_sidl_array.h

# most of the rest of the file should not require editing

ifeq ($(IMPLSRCS),)
  SCLFILE=
  BABELFLAG=--client=Chapel
  MODFLAG=
else
  SCLFILE=lib$(LIBNAME).scl
  BABELFLAG=--server=Chapel
  MODFLAG=-module
  DCE=--no-dead-code-elimination # include everything in libimpl.la
endif

ifeq ($(CHAPEL_MAKE_COMM),gasnet)

all: lib$(LIBNAME).la $(SCLFILE) $(OUTFILE) $(OUTFILE)_real

# actual program
$(OUTFILE)_real: lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
#	babel-libtool --mode=link --tag=CXX /usr/local/bin/mpicc -static lib$(LIBNAME).la 
#	babel-libtool --mode=link --tag=CXX $(CXX) -static lib$(LIBNAME).la 
	babel-libtool --mode=link --tag=CXX /usr/local/bin/mpicc -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) \
          $(CHPL_GASNET_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

# launcher
$(OUTFILE): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo
	echo "#include \"chplcgfns.h\"" > $(IMPL).chpl.dir/config.c
	echo "#include \"config.h\""   >> $(IMPL).chpl.dir/config.c
	echo "#include \"_config.c\""  >> $(IMPL).chpl.dir/config.c
	babel-libtool --mode=compile --tag=CC $(CC) \
          -std=c99 -I$(CHAPEL_ROOT)/runtime/include/$(CHPL_HOST_PLATFORM) \
	  -I$(CHAPEL_ROOT)/runtime/include -I. \
	  $(IMPL).chpl.dir/config.c -c -o $@.lo
	babel-libtool --mode=link $(CC) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $@.lo $(SERVER) \
          $(CHPL_LAUNCHER_LDFLAGS) $(LAUNCHER_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

else

all: lib$(LIBNAME).la $(SCLFILE) $(OUTFILE)

$(OUTFILE): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
	babel-libtool --mode=link $(CC) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) $(CHPL_LDFLAGS) $(EXTRA_LDFLAGS) -o $@
endif

CC=`babel-config --query-var=CC`
INCLUDES=`babel-config --includes` -I. -I$(CHAPEL_ROOT)/runtime/include -I$(SIDL_RUNTIME)/chpl
CFLAGS=`babel-config --flags-c` -std=c99
LIBS=`babel-config --libs-c-client`

STUBOBJS=$(patsubst .chpl, .lo, $(STUBSRCS:.c=.lo))
IOROBJS=$(IORSRCS:.c=.lo)
SKELOBJS=$(SKELSRCS:.c=.lo)
IMPLOBJS=$(IMPLSRCS:.chpl=.lo)

PUREBABELGEN=$(IORHDRS) $(IORSRCS) $(STUBSRCS) $(STUBHDRS) $(SKELSRCS)
BABELGEN=$(IMPLHDRS) $(IMPLSRCS)

$(IMPLOBJS) : $(STUBHDRS) $(IORHDRS) $(IMPLHDRS)

lib$(LIBNAME).la : $(STUBOBJS) $(IOROBJS) $(IMPLOBJS) $(SKELOBJS)
	babel-libtool --mode=link --tag=CC $(CC) -o lib$(LIBNAME).la \
	  -rpath $(LIBDIR) -release $(VERSION) \
	  -no-undefined $(MODFLAG) \
	  $(CFLAGS) $(EXTRAFLAGS) $^ $(LIBS) \
	  $(EXTRALIBS)

$(PUREBABELGEN) $(BABELGEN) : babel-stamp
	@if test -f $@; then \
	    touch $@; \
	else \
	    rm -f babel-stamp ; \
	    $(MAKE) babel-stamp; \
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
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; \
	   fi ; \
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
	$(CHPL) --savec $<.dir $< *Stub.h $(CHPL_HEADERS) $(DCE) --make true  # gen C-code only
	babel-libtool --mode=compile --tag=CC $(CC) \
            -I./$<.dir $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) \
            $(CHPL_FLAGS) -c -o $@ $<.dir/_main.c
else
.chpl.lo:
	$(CHPL) --savec $<.dir $< *Stub.h $(CHPL_HEADERS) $(DCE) --make true  # gen C-code
	headerize $<.dir/_config.c $<.dir/Chapel*.c $<.dir/Default*.c $<.dir/DSIUtil.c $<.dir/chpl*.c $<.dir/List.c $<.dir/Math.c $<.dir/Search.c $<.dir/Sort.c $<.dir/Types.c
	perl -pi -e 's/((chpl__autoDestroyGlobals)|(chpl_user_main)|(chpl__init)|(chpl_main))/$*_\1/g' $<.dir/$*.c
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
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; \
	   fi ; \
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
