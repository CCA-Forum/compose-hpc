#!/bin/bash
DIR=/home/prantl1/work/babel/braid.build/t
#SIDLFILE=../../shams/example-5/DistrArray.sidl
SIDLFILE=../../compose-hpc/braid/regression/parser/hello.sidl
rm -rf $DIR/*
cd $DIR
cat >impl.chpl <<EOF
module HelloWorld {
    use hello;
    proc main() {
        var hw: hello.World;
        hw = new hello.World();
        hw.setName("Hello World!");
        writeln(hw.getMsg());
        delete hw;
    }
}
EOF

make -C .. && make -C .. install >/dev/null && \
braid $SIDLFILE -cchapel --makefile && \
mkdir libC && cd libC && \
echo babel ../$SIDLFILE -sc --makefile && \
babel ../$SIDLFILE -sc --makefile && \
perl -pi -e"s/(EXTRAFLAGS=)/\1-ggdb -O0/" GNUmakefile && \
env PYTHONPATH=/home/prantl1/work/babel/install/lib/python2.6/site-packages/braid python <<EOF 
import splicer
splicer.replace('hello_World_Impl.c', 'hello.World.getMsg', 
                'return "Hello from C!";\n') 
EOF
make -j8 &>/dev/null && cd .. && \
make SERVER=libC/libimpl IMPL=impl && \
./runChapel |grep 'Hello from C!'
