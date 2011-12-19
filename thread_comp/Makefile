#HWLOC_HOME = /project/projectdirs/m876/foley/hwloc-1.1.1
HWLOC_HOME = /usr/lib64/hwloc
#HWLOC_HOME = /project/projectdirs/m876/foley/hwloc-1.1.1
#HWLOC_HOME = /u/sfoley/hwloc
#HWLOC_HOME = /u/ssfoley

SCALAPACK_HOME = /usr/lib64/scalapack/openmpi-1.4.3/scalapack-1.8.0
BLACS_HOME = /usr/lib64/BLACS

#MPICC = cc
MPICC = mpicc
OMP_FLAGS = -fopenmp
MPIF = mpif77 -fopenmp

CFLAGS = -I$(HWLOC_HOME)/include
LDLIBS = -L$(HWLOC_HOME)/lib -lhwloc

ACML_LIBS = /usr/lib64/acml4.4.0/gfortran64/lib/libacml.a
ACML_MP_LIBS = /usr/lib64/acml4.4.0/gfortran64_mp/lib/libacml_mp.a /usr/lib64/acml4.4.0/gfortran64_mp/lib/libacml_mv.a
ACML_HOME = /usr/lib64/acml4.4.0/gfortran64
ACML_HOME_MP = /usr/lib64/acml4.4.0/gfortran64_mp

# flags are the locations of the relevant .h files
SCALA_FLAGS = -I$(SCALAPACK_HOME)/SRC -I$(BLACS_HOME)/SRC/MPI -I$(ACML_HOME)/lib -DAdd_ -DBlacsDebugLvl=1 -DNO_IEEE -DUseMpi2 -O4 -DBTCINTFACE -DSYSINC
SCALA_LDLIBS = $(SCALAPACK_HOME)/libscalapack.a $(BLACS_HOME)/LIB/blacsCinit_MPI-LINUX-1.a $(BLACS_HOME)/LIB/blacs_MPI-LINUX-1.a $(BLACS_HOME)/LIB/blacsCinit_MPI-LINUX-1.a $(ACML_LIBS)

SCALA_MP_FLAGS = -I$(SCALAPACK_HOME)/SRC -I$(BLACS_HOME)/SRC/MPI -I$(ACML_HOME_MP)/lib -DAdd_ -DBlacsDebugLvl=1 -DNO_IEEE -DUseMpi2 -O4 -DBTCINTFACE -DSYSINC
SCALA_MP_LDLIBS = $(SCALAPACK_HOME)/libscalapack.a $(BLACS_HOME)/LIB/blacsCinit_MPI-LINUX-1.a $(BLACS_HOME)/LIB/blacs_MPI-LINUX-1.a $(BLACS_HOME)/LIB/blacsCinit_MPI-LINUX-1.a $(ACML_MP_LIBS)

#CFLAGS += $(pkg-config --cflags hwloc)
#LDLIBS += $(pkg-config --libs hwloc)

topo: topo_disco topo_check

hwloc-hello:
	$(MPICC) hwloc-hello.c $(CFLAGS) -o hwloc-hello $(LDLIBS)

topo_disco:
	$(MPICC) topo_disco.c $(CFLAGS) -o topo_disco $(LDLIBS)

topo_check:
	$(MPICC) topo_check.c $(CFLAGS) -o topo_check $(LDLIBS)

topo_check_long:
	$(MPICC) topo_check_long.c $(CFLAGS) -o topo_check_long $(LDLIBS)

hw:
	$(MPICC) -o hw hw.c

mixed_hw:
	$(MPICC) $(OMP_FLAGS) mixed_hw.c -o mixed_hw

mixed_blacs_hw:
	$(MPICC) $(OMP_FLAGS) mixed_blacs_hw.c -o mixed_blacs_hw

mixed_context:
	$(MPICC) -c $(OMP_FLAGS) $(SCALA_MP_FLAGS) mixed_context.c 
	$(MPIF) -o mixed_context mixed_context.o $(SCALA_MP_LDLIBS)

mixed_skel:
	$(MPICC) -c $(OMP_FLAGS) $(SCALA_MP_FLAGS) mixed_skel.c 
	$(MPIF) -o mixed_skel mixed_skel.o $(SCALA_MP_LDLIBS)

scala_solver:
	$(MPICC) -c $(OMP_FLAGS) $(SCALA_MP_FLAGS) scala_solver.c 
	$(MPIF) -o scala_solver scala_solver.o $(SCALA_MP_LDLIBS)

hw_long:
	$(MPICC) -o hw_long hw_long.c

hw-cray:
	$(MPICC) -fast -o hw hw.c	

clean:
	rm -rf hwloc-hello topo_disco topo_check topo_check_long hw mixed_context mixed_skel scala_solver hw_long *.o
