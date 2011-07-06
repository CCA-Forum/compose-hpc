#!/bin/bash

rm -f *.o
rm -f *.lo
rm -f a.out
rm -rf gen

CC="`babel-config --query-var=CC`"
CXX="`babel-config --query-var=CXX`"

CHAPEL_HOME="${CHPL_HOME}"
CHAPEL_COMM="${CHPL_COMM}"
CHAPEL_HOST_PLATFORM="${CHPL_HOST_PLATFORM}"
CHAPEL_SUBSTRATE_DIR="${CHPL_HOME}/lib/${CHPL_HOST_PLATFORM}/gnu/comm-none/substrate-none"
if [ "$CHPL_COMM" = "gasnet" ] ; then
  CHAPEL_SUBSTRATE_DIR="${CHAPEL_HOME}/lib/${CHAPEL_HOST_PLATFORM}/gnu/comm-gasnet-nodbg/substrate-udp"
fi
CHAPEL_RUNTIME_INC_DIR="${CHPL_HOME}/runtime/include"

SIDL_RUNTIME="/Users/imam1/softwares/include"
INCLUDES="`babel-config --includes` -I. -I${CHAPEL_RUNTIME_INC_DIR} -I${SIDL_RUNTIME}/chpl"
CFLAGS="`babel-config --flags-c` -std=c99"
LIBS="`babel-config --libs-c-client`"

COMM_INCLUDE="${CHAPEL_RUNTIME_INC_DIR}/comm/none"
if [ "$CHPL_COMM" = "gasnet" ] ; then
  COMM_INCLUDE="${CHAPEL_RUNTIME_INC_DIR}/comm/gasnet"
fi
GASNET_FLAGS=""
if [ "$CHPL_COMM" = "gasnet" ] ; then
  GASNET_FLAGS="-no-cpp-precomp   -DGASNET_PAR -D_REENTRANT   -I${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/include  -I${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/include/udp-conduit"
fi

CHPL_FLAGS="-D_POSIX_C_SOURCE  -DCHPL_TASKS_H=\"tasks-fifo.h\"  -DCHPL_THREADS_H=\"threads-pthreads.h\" -I${CHAPEL_RUNTIME_INC_DIR}/tasks/fifo  -I${CHAPEL_RUNTIME_INC_DIR}/threads/pthreads -I${COMM_INCLUDE} -I${CHAPEL_RUNTIME_INC_DIR}/comp-gnu -I${CHAPEL_RUNTIME_INC_DIR}/${CHAPEL_HOST_PLATFORM} -I${CHAPEL_RUNTIME_INC_DIR} -I. -Wno-all"

CHPL_LDFLAGS="-L${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads ${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads/main.o -lchpl  -lm  -lpthread"
GASNET_LDFLAGS=""
if [ "$CHPL_COMM" = "gasnet" ] ; then
  GASNET_LDFLAGS="-L${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/lib  -lgasnet-udp-par -lamudp -lgcc "
fi

EXTRA_LDFLAGS=""

# extra include/compile flags
EXTRAFLAGS="-ggdb -O0"
# extra libraries that the implementation needs to link against
EXTRALIBS=
# PREFIX specifies the top of the installation directory
PREFIX=/usr/local
# the default installation installs the .la and .scl (if any) into the LIBDIR
LIBDIR=${PREFIX}/lib

BABEL_LIBTOOL_COMMAND="babel-libtool --mode=compile --tag=CC ${CC} -I./gen ${INCLUDES} ${CFLAGS} ${EXTRAFLAGS} ${GASNET_FLAGS} ${CHPL_FLAGS}"

# Generate the C source files for the chapel program
# chpl --savec ./gen *.chpl 
chpl --savec ./gen *.chpl --make true 

# Compile the babel generated C source files and generate the object files

${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_BlockCyclicDistArray2dDouble_IOR.c.o     hplsupport_BlockCyclicDistArray2dDouble_IOR.c
${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_BlockCyclicDistArray2dDouble_Stub.c.o    hplsupport_BlockCyclicDistArray2dDouble_Stub.c
${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_BlockCyclicDistArray2dDouble_Skel.c.o    hplsupport_BlockCyclicDistArray2dDouble_Skel.c
${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_BlockCyclicDistArray2dDouble_cImpl.c.o   hplsupport_BlockCyclicDistArray2dDouble_cImpl.c

${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_SimpleArray1dInt_IOR.c.o     hplsupport_SimpleArray1dInt_IOR.c
${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_SimpleArray1dInt_Stub.c.o    hplsupport_SimpleArray1dInt_Stub.c
${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_SimpleArray1dInt_Skel.c.o    hplsupport_SimpleArray1dInt_Skel.c
${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_SimpleArray1dInt_cImpl.c.o   hplsupport_SimpleArray1dInt_cImpl.c

# Compile the client C source and generate the object file
${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/hplsupport_HPL_cClient.c.o  hplsupport_HPL_cClient.c

# Compile the chapel generated C source files and generate the object files
if [ "$CHPL_COMM" = "gasnet" ] ; then

    ${BABEL_LIBTOOL_COMMAND}  -c  -o ./gen/a.out.tmp.o   ./gen/_main.c 

else

  echo "CHPL_COMM is NOT gasnet"
  ${BABEL_LIBTOOL_COMMAND} -c  -o ./gen/a.out.tmp.o   ./gen/_main.c

fi

 
MODFLAG=""
# MODFLAG="module"            

# link all the object files
# ${CC_COMMAND}   -o ./gen/a.out.tmp  ./gen/a.out.tmp.o ${CHPL_LDFLAGS}
babel-libtool --mode=link ${CXX} -static \
  -o ./gen/a.out.tmp  \
  -rpath ${LIBDIR} \
  -no-undefined ${MODFLAG} \
  ${CFLAGS} ${EXTRAFLAGS} ${LIBS} \
  ${EXTRALIBS} \
  ${CHPL_LDFLAGS} ${GASNET_LDFLAGS} ${EXTRA_LDFLAGS} \
  ./gen/hplsupport_BlockCyclicDistArray2dDouble_IOR.c.o \
  ./gen/hplsupport_BlockCyclicDistArray2dDouble_Stub.c.o \
  ./gen/hplsupport_BlockCyclicDistArray2dDouble_Skel.c.o \
  ./gen/hplsupport_BlockCyclicDistArray2dDouble_cImpl.c.o \
  ./gen/hplsupport_SimpleArray1dInt_IOR.c.o \
  ./gen/hplsupport_SimpleArray1dInt_Stub.c.o \
  ./gen/hplsupport_SimpleArray1dInt_Skel.c.o \
  ./gen/hplsupport_SimpleArray1dInt_cImpl.c.o \
  ./gen/hplsupport_HPL_cClient.c.o \
  ./gen/a.out.tmp.o 
    
    
if [ "$CHPL_COMM" = "gasnet" ] ; then

  cp ./gen/a.out.tmp a.out_real
  rm ./gen/a.out.tmp
  echo "#include \"chplcgfns.h\"" > ./gen/config.c
  echo "#include \"config.h\"" >> ./gen/config.c
  echo "#include \"_config.c\"" >> ./gen/config.c

  ${CC} -std=c99 -D_POSIX_C_SOURCE   -c -o ./gen/a.out.tmp_launcher.o -I${CHAPEL_RUNTIME_INC_DIR}/${CHAPEL_HOST_PLATFORM} -I${CHAPEL_RUNTIME_INC_DIR} -I. ./gen/config.c
  ${CC}   -o ./gen/a.out.tmp_launcher -L${CHAPEL_SUBSTRATE_DIR}/launch-amudprun ./gen/a.out.tmp_launcher.o ${CHAPEL_SUBSTRATE_DIR}/launch-amudprun/main_launcher.o -lchpllaunch -lm 

  cp ./gen/a.out.tmp_launcher ./gen/a.out.tmp
  cp ./gen/a.out.tmp a.out
  rm ./gen/a.out.tmp


else

  cp ./gen/a.out.tmp a.out
  rm ./gen/a.out.tmp

fi



