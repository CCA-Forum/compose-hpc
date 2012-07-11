export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=ibv
make -j12 -C ~/sw/opt/chapel/ clean
make CC='mpicc -cc=gcc-4.3' CXX='mpic++ -cxx=g++-4.3' -j12 -C ~/sw/opt/chapel/ depend

make CC='mpicc -cc=gcc-4.3' CXX='mpic++ -cxx=g++-4.3' CFLAGS='-O2 -g -march=native' CXXFLAGS='-O2 -g -march=native' -C ~/sw/opt/chapel/

