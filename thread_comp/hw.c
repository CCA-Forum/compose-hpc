#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
    int my_rank;
    int p;
    int source;
    int dest;
    int tag = 0;
    char message[100];
    char hname[50];
    pid_t my_pid;
    MPI_Status status;

    //Start up MPI
    MPI_Init(&argc, &argv);

    //find process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //find number of procs
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // all procs get hostname and pid
    gethostname(hname, 50);
    my_pid = getpid();


    if(my_rank != 0){
        //create message
        sprintf(message, "Greetings from process %d (pid %d host %s)!", my_rank, my_pid, hname);
        dest = 0;
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    }
    else{  //my rank is 0        
        printf("Greetings from process %d (pid %d host %s)!\n", my_rank, my_pid, hname);
        for(source = 1; source < p; source++){
            MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
            printf("%s\n", message);
        }
    }

    //shutdown MPI
    MPI_Finalize();

    return 0;
}
