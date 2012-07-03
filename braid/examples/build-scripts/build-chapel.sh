make -j12 -C ~/sw/opt/chapel/ clean
#make -j12 -C ~/sw/opt/chapel/ depend

make CC='mpicc -cc=gcc-4.3' CXX='mpig++ -cc=g++-4.3' CFLAGS='-O2 -g -march=native' CXXFLAGS='-O2 -g -march=native' -C ~/sw/opt/chapel/

