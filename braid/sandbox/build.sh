#!/bin/bash

rm *.o

rm a.out

rm -rf csource

GCC_COMMAND=/nfs/apps/gcc/4.3.2/bin/gcc
GCPP_COMMAND=/nfs/apps/gcc/4.3.2/bin/g++

CHAPEL_HOME="${CHPL_HOME}"
CHAPEL_COMM="${CHPL_COMM}"

echo "chapel comm = ${CHAPEL_COMM}"

chpl --savec ./csource DistArray_Client.h DistArray.chpl --make true 

if [ CHAPEL_COMM ]
then

    echo "Compiling for gasnet mode"

    CHAPEL_SUBSTRATE_DIR=${CHPL_HOME}/lib/linux32/gnu/comm-gasnet-nodbg/substrate-udp
    CHAPEL_GASNET_NODEBG_DIR=${CHAPEL_HOME}/third-party/gasnet/install/linux32-gnu/seg-everything/nodbg

    ${GCC_COMMAND}  -c -std=c99  -I. -I./csource  -o DistArray_Stub.c.o    DistArray_Stub.c
    ${GCC_COMMAND}  -c -std=c99  -I. -I./csource  -o DistArray_Client.c.o  DistArray_Client.c

    ${GCC_COMMAND}    -DGASNET_PAR -D_REENTRANT -D_GNU_SOURCE   -I${CHAPEL_GASNET_NODEBG_DIR}/include -I${CHAPEL_GASNET_NODEBG_DIR}/include/udp-conduit      -std=c99 -DCHPL_TASKS_H=\"tasks-fifo.h\" -DCHPL_THREADS_H=\"threads-pthreads.h\"   -c -o ./csource/a.out.tmp.o -I${CHAPEL_HOME}/runtime/include/tasks/fifo -I${CHAPEL_HOME}/runtime/include/threads/pthreads -I${CHAPEL_HOME}/runtime/include/comm/gasnet -I${CHAPEL_HOME}/runtime/include/comp-gnu -I${CHAPEL_HOME}/runtime/include/linux32 -I${CHAPEL_HOME}/runtime/include -I. ./csource/_main.c 

    ${GCPP_COMMAND}     -o ./csource/a.out.tmp -L${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads ./csource/a.out.tmp.o ${CHAPEL_SUBSTRATE_DIR}/tasks-fifo/threads-pthreads/main.o DistArray_Stub.c.o DistArray_Client.c.o  -lchpl -lm  -lpthread -L${CHAPEL_GASNET_NODEBG_DIR}/lib   -lgasnet-udp-par -lamudp     -lpthread -L/nfs/apps/gcc/4.3.2/lib/gcc/i686-pc-linux-gnu/4.3.2 -lgcc -lm 

    gmake -f ${CHAPEL_HOME}/runtime/etc/Makefile.launcher all CHAPEL_ROOT=${CHAPEL_HOME} TMPBINNAME=./csource/a.out.tmp BINNAME=a.out TMPDIRNAME=./csource

    cp ./csource/a.out.tmp a.out_real
    rm ./csource/a.out.tmp

    echo "#include \"chplcgfns.h\"" > ./csource/config.c
    echo "#include \"config.h\"" >> ./csource/config.c
    echo "#include \"_config.c\"" >> ./csource/config.c

    ${GCC_COMMAND} -std=c99  -c -o ./csource/a.out.tmp_launcher.o -I${CHAPEL_HOME}/runtime/include/linux32 -I${CHAPEL_HOME}/runtime/include -I. ./csource/config.c

    ${GCC_COMMAND}   -o ./csource/a.out.tmp_launcher -L${CHAPEL_SUBSTRATE_DIR}/launch-amudprun ./csource/a.out.tmp_launcher.o ${CHAPEL_SUBSTRATE_DIR}/launch-amudprun/main_launcher.o -lchpllaunch -lm 

    cp ./csource/a.out.tmp_launcher ./csource/a.out.tmp
    cp ./csource/a.out.tmp a.out
    rm ./csource/a.out.tmp

else

    echo "Compiling for non-gasnet mode"

    ${GCC_COMMAND} -c -std=c99  -I. -I./csource  -o DistArray_Stub.c.o    DistArray_Stub.c
    ${GCC_COMMAND} -c -std=c99  -I. -I./csource  -o DistArray_Client.c.o  DistArray_Client.c

    ${GCC_COMMAND} -std=c99 -DCHPL_TASKS_H=\"tasks-fifo.h\" -DCHPL_THREADS_H=\"threads-pthreads.h\"   -c -o ./csource/a.out.tmp.o -I${CHAPEL_HOME}/runtime/include/tasks/fifo -I${CHAPEL_HOME}/runtime/include/threads/pthreads -I${CHAPEL_HOME}/runtime/include/comm/none -I${CHAPEL_HOME}/runtime/include/comp-gnu -I${CHAPEL_HOME}/runtime/include/linux32 -I${CHAPEL_HOME}/runtime/include -I. ./csource/_main.c 

    ${GCC_COMMAND}   -o ./csource/a.out.tmp -L${CHAPEL_HOME}/lib/linux32/gnu/comm-none/substrate-none/tasks-fifo/threads-pthreads ./csource/a.out.tmp.o ${CHAPEL_HOME}/lib/linux32/gnu/comm-none/substrate-none/tasks-fifo/threads-pthreads/main.o DistArray_Stub.c.o DistArray_Client.c.o -lchpl -lm  -lpthread

    cp ./csource/a.out.tmp a.out
    rm ./csource/a.out.tmp

fi





