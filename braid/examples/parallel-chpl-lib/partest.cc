#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include "partest.hxx"
#include <unistd.h>

using namespace std;
extern char* babel_program_name;

int main(int argc, char *argv[])
{
  int numProcs ;
  int myRank ;
  MPI_Init(&argc, &argv) ;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
  cout<<"["<<getenv("SLURM_NODEID")<<": "<<getenv("HOSTNAME")<<"] "
      <<"numProcs="<<numProcs<<", myRank="<<myRank<<endl;

  cout<<"["<<getenv("SLURM_NODEID")<<": "<<getenv("HOSTNAME")<<"] "
      <<"1 before Chapel runtime initialization"<<endl;

  partest::Hello hello = partest::Hello::_create();

  cout<<"["<<getenv("SLURM_NODEID")<<": "<<getenv("HOSTNAME")<<"] "
      <<"2 after Chapel runtime initialization"<<endl;

  if (myRank == 0) {
    hello.sayHello();
    cout<<"["<<getenv("SLURM_NODEID")<<": "<<getenv("HOSTNAME")<<"] "
	<<"3 returned from Chapel call"<<endl;
  }

  // MPI_Finalize() ;

  return 0 ;
}
