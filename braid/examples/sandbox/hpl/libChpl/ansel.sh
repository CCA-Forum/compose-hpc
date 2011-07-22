#!/bin/bash

CC="`babel-config --query-var=CC`" # "colorgcc.pl"
CXX="/usr/local/bin/mpicc"

CHAPEL_HOME="${CHPL_HOME}"
CHAPEL_COMM="${CHPL_COMM}"
CHAPEL_HOST_PLATFORM="${CHPL_HOST_PLATFORM}"
CHAPEL_SUBSTRATE_DIR="${CHAPEL_HOME}/lib/${CHAPEL_HOST_PLATFORM}/gnu/comm-gasnet-nodbg/substrate-ibv"
CHAPEL_RUNTIME_INC_DIR="${CHPL_HOME}/runtime/include"
CHAPEL_COMM_INCLUDE="${CHAPEL_RUNTIME_INC_DIR}/comm/gasnet"

SIDL_RUNTIME="/g/g91/prantl1/install/include"
INCLUDES="`babel-config --includes` -I. -I${CHAPEL_RUNTIME_INC_DIR} -I${SIDL_RUNTIME}/chpl  -I${CHAPEL_RUNTIME_INC_DIR}/tasks/fifo  -I${CHAPEL_RUNTIME_INC_DIR}/threads/pthreads -I${CHAPEL_COMM_INCLUDE} -I${CHAPEL_RUNTIME_INC_DIR}/comp-gnu -I${CHAPEL_RUNTIME_INC_DIR}/${CHAPEL_HOST_PLATFORM} -I${CHAPEL_RUNTIME_INC_DIR} -I."
CFLAGS="-std=c99"
LIBS="`babel-config --libs-c-client`"

GASNET_FLAGS="-DGASNET_PAR -D_REENTRANT -DGNU_SOURCE   -I${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/include  -I${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/include/ibv-conduit -DGASNET_CONDUIT_IBV"
CHPL_FLAGS="-DCHPL_TASKS_H=\"tasks-fifo.h\"  -DCHPL_THREADS_H=\"threads-pthreads.h\""
CHPL_LDFLAGS="-L${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads ${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads/main.o -lchpl  -lm  -lpthread"
GASNET_LDFLAGS="-L${CHAPEL_HOME}/third-party/gasnet/install/${CHAPEL_HOST_PLATFORM}-gnu/seg-everything/nodbg/lib  -lgasnet-ibv-par -libverbs -lgcc "

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
HEADER_DEPS="${HEADER_DEPS} hpcc.h"
HEADER_DEPS="${HEADER_DEPS} hplsupport.h"
HEADER_DEPS="${HEADER_DEPS} braid_chapel_util.h"
HEADER_DEPS="${HEADER_DEPS} hplsupport_SimpleArray1dInt_cStub.h"
HEADER_DEPS="${HEADER_DEPS} hplsupport_BlockCyclicDistArray2dDouble_cStub.h"

BRAID_GEN_C_SOURCES=""

BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_BlockCyclicDistArray2dDouble_IOR.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_BlockCyclicDistArray2dDouble_Stub.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_BlockCyclicDistArray2dDouble_cStub.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_BlockCyclicDistArray2dDouble_Skel.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_BlockCyclicDistArray2dDouble_cImpl.c"

BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_SimpleArray1dInt_IOR.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_SimpleArray1dInt_Stub.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_SimpleArray1dInt_cStub.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_SimpleArray1dInt_Skel.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_SimpleArray1dInt_cImpl.c"

BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hpcc_HighPerformanceLinpack_IOR.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hpcc_HighPerformanceLinpack_Stub.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hpcc_HighPerformanceLinpack_Skel.c"
BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hpcc_HighPerformanceLinpack_cImpl.c"

# BRAID_GEN_C_SOURCES="${BRAID_GEN_C_SOURCES} hplsupport_HPL_cClient.c "

BRAID_GEN_O_FILES=""




clear;
set -e 

echo "Cleaning previous build artifacts"
rm -f *.o; rm -f *.lo; rm -f runHpl*; rm -rf gen;

echo "Generating C files from chpl files"
chpl --savec ./gen ${HEADER_DEPS} *.chpl --make true

for loopFile in ${BRAID_GEN_C_SOURCES}
do
  echo "Compiling ${loopFile}"
  ${BABEL_LIBTOOL_COMMAND}  -c  -o ./gen/${loopFile}.o  ${loopFile}
  BRAID_GEN_O_FILES="${BRAID_GEN_O_FILES} ./gen/${loopFile}.o"
done

echo "Compiling ./gen/_main.c"
echo "BRAID_GEN_O_FILES = ${BRAID_GEN_O_FILES}"
${BABEL_LIBTOOL_COMMAND}  -c  -o ./gen/runHpl.tmp.o  ./gen/_main.c 

echo "Linking all files"
babel-libtool --tag=CXX --mode=link ${CXX} -static \
  -o ./gen/runHpl.tmp  \
  -rpath ${LIBDIR} \
  ${CFLAGS} ${EXTRAFLAGS} ${LIBS} \
  ${EXTRALIBS} \
  ${CHPL_LDFLAGS} ${GASNET_LDFLAGS} ${EXTRA_LDFLAGS} \
  ${BRAID_GEN_O_FILES} \
  ./gen/runHpl.tmp.o 
  
echo "Generating launcher"

echo " Creating runHpl_real"
cp ./gen/runHpl.tmp runHpl_real
rm ./gen/runHpl.tmp

echo " Creating config.c"
echo "#include \"chplcgfns.h\"" > ./gen/config.c
echo "#include \"config.h\"" >> ./gen/config.c
echo "#include \"_config.c\"" >> ./gen/config.c

echo " Compiling config.c"
${CC} -std=c99 -D_POSIX_C_SOURCE -c -o ./gen/runHpl.tmp_launcher.o -I${CHAPEL_RUNTIME_INC_DIR}/${CHAPEL_HOST_PLATFORM} -I${CHAPEL_RUNTIME_INC_DIR} -I. ./gen/config.c 

echo " Linking the launcher"
${CC}  -o ./gen/runHpl.tmp_launcher -L${CHAPEL_SUBSTRATE_DIR}/launch-gasnetrun_ibv ./gen/runHpl.tmp_launcher.o ${CHAPEL_SUBSTRATE_DIR}/launch-gasnetrun_ibv/main_launcher.o -lchpllaunch -lm 

echo " Creating runHpl"
cp ./gen/runHpl.tmp_launcher ./gen/runHpl.tmp
cp ./gen/runHpl.tmp runHpl
rm ./gen/runHpl.tmp  

  
