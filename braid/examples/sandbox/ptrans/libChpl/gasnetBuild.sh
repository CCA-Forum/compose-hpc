#!/bin/bash

CC="`babel-config --query-var=CC`"
CXX="`babel-config --query-var=CXX`"

CHAPEL_HOME="${CHPL_HOME}"
CHAPEL_COMM="${CHPL_COMM}"
CHAPEL_HOST_PLATFORM="${CHPL_HOST_PLATFORM}"
CHAPEL_SUBSTRATE_DIR="${CHAPEL_HOME}/lib/${CHAPEL_HOST_PLATFORM}/gnu/comm-gasnet-nodbg/substrate-udp"
CHAPEL_RUNTIME_INC_DIR="${CHPL_HOME}/runtime/include"
CHAPEL_COMM_INCLUDE="${CHAPEL_RUNTIME_INC_DIR}/comm/gasnet"

SIDL_RUNTIME="/Users/imam1/softwares/include"
INCLUDES="`babel-config --includes` -I. -I${CHAPEL_RUNTIME_INC_DIR} -I${SIDL_RUNTIME}/chpl  -I${CHAPEL_RUNTIME_INC_DIR}/tasks/fifo  -I${CHAPEL_RUNTIME_INC_DIR}/threads/pthreads -I${CHAPEL_COMM_INCLUDE} -I${CHAPEL_RUNTIME_INC_DIR}/comp-gnu -I${CHAPEL_RUNTIME_INC_DIR}/${CHAPEL_HOST_PLATFORM} -I${CHAPEL_RUNTIME_INC_DIR} -I."
CFLAGS="-std=c99"
LIBS="`babel-config --libs-c-client`"

GASNET_FLAGS="-no-cpp-precomp   -DGASNET_PAR -D_REENTRANT   -I${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/include  -I${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/include/udp-conduit"
CHPL_FLAGS="-D_POSIX_C_SOURCE  -DCHPL_TASKS_H=\"tasks-fifo.h\"  -DCHPL_THREADS_H=\"threads-pthreads.h\""
CHPL_LDFLAGS="-L${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads ${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads/main.o -lchpl  -lm  -lpthread"
GASNET_LDFLAGS="-L${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/lib  -lgasnet-udp-par -lamudp -lgcc "

EXTRA_LDFLAGS=""

# extra include/compile flags
EXTRAFLAGS=""

# extra libraries that the implementation needs to link against
EXTRALIBS=""

# PREFIX specifies the top of the installation directory
PREFIX="/usr/local"
# the default installation installs the .la and .scl (if any) into the LIBDIR
LIBDIR="${PREFIX}/lib"

BABEL_LIBTOOL_COMMAND="babel-libtool --mode=compile --tag=CC ${CC} ${CFLAGS} ${EXTRAFLAGS} ${GASNET_FLAGS} ${CHPL_FLAGS} -I./gen ${INCLUDES}"


HEADER_DEPS=""
HEADER_DEPS="${HEADER_DEPS} hplsupport_BlockCyclicDistArray2dDouble.h"
HEADER_DEPS="${HEADER_DEPS} hplsupport_BlockCyclicDistArray2dDouble_cStub.h"
HEADER_DEPS="${HEADER_DEPS} braid_chapel_util.h"

BRAID_GEN_C_SOURCES="hplsupport_BlockCyclicDistArray2dDouble_IOR.c hplsupport_BlockCyclicDistArray2dDouble_Skel.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_BlockCyclicDistArray2dDouble_Stub.c hplsupport_BlockCyclicDistArray2dDouble_cStub.c "
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_BlockCyclicDistArray2dDouble_cImpl.c "
# BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_HPL_cClient.c "

BRAID_GEN_O_FILES=""




clear

echo "Cleaning up previous build artifacts"
rm -f *.o; rm -f *.lo; rm -f a.out*; rm -rf gen;

echo "Generating C-sources from chpl files"
chpl --fast --savec ./gen  ${HEADER_DEPS} *.chpl --make true

for loopFile in ${BRAID_GEN_C_SOURCES}
do
  echo "Compiling ${loopFile}"
  ${BABEL_LIBTOOL_COMMAND}  -c  -o ./gen/${loopFile}.o  ${loopFile}
  BRAID_GEN_O_FILES="${BRAID_GEN_O_FILES} ./gen/${loopFile}.o"
done

echo "Compiling ./gen/_main.c"
echo "BRAID_GEN_O_FILES = ${BRAID_GEN_O_FILES}"
${BABEL_LIBTOOL_COMMAND}  -c  -o ./gen/a.out.tmp.o  ./gen/_main.c 

echo "Linking all files"
babel-libtool --mode=link ${CXX} -static \
  -o ./gen/a.out.tmp  \
  -rpath ${LIBDIR} \
  ${CFLAGS} ${EXTRAFLAGS} ${LIBS} \
  ${EXTRALIBS} \
  ${CHPL_LDFLAGS} ${GASNET_LDFLAGS} ${EXTRA_LDFLAGS} \
  ${BRAID_GEN_O_FILES} \
  ./gen/a.out.tmp.o 
  
echo "Generating launcher"

echo " Creating a.out_real"
cp ./gen/a.out.tmp a.out_real
rm ./gen/a.out.tmp

echo " Creating config.c"
echo "#include \"chplcgfns.h\"" > ./gen/config.c
echo "#include \"config.h\"" >> ./gen/config.c
echo "#include \"_config.c\"" >> ./gen/config.c

echo " Compiling config.c"
${CC} -std=c99 -D_POSIX_C_SOURCE -c -o ./gen/a.out.tmp_launcher.o -I${CHAPEL_RUNTIME_INC_DIR}/${CHAPEL_HOST_PLATFORM} -I${CHAPEL_RUNTIME_INC_DIR} -I. ./gen/config.c 

echo " Linking the launcher"
${CC}  -o ./gen/a.out.tmp_launcher -L${CHAPEL_SUBSTRATE_DIR}/launch-amudprun ./gen/a.out.tmp_launcher.o ${CHAPEL_SUBSTRATE_DIR}/launch-amudprun/main_launcher.o -lchpllaunch -lm 

echo " Creating a.out"
cp ./gen/a.out.tmp_launcher ./gen/a.out.tmp
cp ./gen/a.out.tmp a.out
rm ./gen/a.out.tmp  

  
