VIZ_CFLAGS=-I/usr/gapps/visit/2.5.1/linux-x86_64-chaos4/include/silo/include/ -DVIZ_MESH=1
VIZ_LDFLAGS=-L/usr/gapps/visit/2.5.1/linux-x86_64/archives/ -lsiloh5 -Wl,-rpath -Wl,/usr/gapps/visit/2.5.1/linux-x86_64-chaos4/lib -lhdf5 -lhdf5_hl

luleshMPI: luleshMPI.o Makefile
	mpig++ $< -o $@ $(VIZ_LDFLAGS)

luleshMPI.o: luleshMPI.cc Makefile
	mpig++ $(VIZ_CFLAGS) -c -O3 -ggdb $< -o $@

clean:
	rm -f luleshMPI luleshMPI.o
