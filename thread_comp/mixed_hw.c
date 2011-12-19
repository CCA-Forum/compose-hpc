#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "mpi.h"

int main(int argc, char* argv[]){
    int my_rank, my_thread_rank;
    int p;
    int source;
    int dest;
    int tag = 0;
    char message[100];
    char hname[50];
    pid_t my_pid;
    MPI_Status status;
    
    // to merge threaded and non-threaded sections
    MPI_Group thread_grp, process_grp;
    MPI_Comm thread_comm;
    int * grp_list;
    int enter_thread_reg;
    int len_grp_list;
    int i;
    
    // for threaded stuff
    int nthreads, tid;
    char msg_list[100][40];
    int len_msg_list;
    int len_msg2;
    char msg2[4000];
    
    //Start up MPI
    MPI_Init(&argc, &argv);
    
    //find process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    //find number of procs
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    MPI_Comm_group(MPI_COMM_WORLD, &process_grp);
    
    // all procs get hostname and pid
    gethostname(hname, 50);
    my_pid = getpid();
    
    omp_set_dynamic(1);
    omp_set_num_threads(1);
    
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

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Elect processes to move into threaded section
    if(my_rank % 4 == 0){
        enter_thread_reg = 1;
    }
    else{
        enter_thread_reg = 0;
    }
    
    MPI_Comm_split(MPI_COMM_WORLD, enter_thread_reg, my_rank, &thread_comm);
    MPI_Comm_rank(thread_comm, &my_thread_rank);
    
    omp_set_num_threads(7);
    // Begin threaded Hello World
    if(enter_thread_reg == 1){
        sprintf(msg2, "\0");
        /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel private(tid) {
        nthreads = omp_get_num_threads();
        len_msg2 = 40 * nthreads;
        
        /* Obtain thread number */
        tid = omp_get_thread_num();
        sprintf(msg_list[tid],"Hi I am tid %d of rank %d (pid %d)\n", tid, my_thread_rank, my_pid);
        //printf(msg_list[tid],"Hi I am tid %d of rank %d\n", tid, my_rank);
#pragma omp barrier
        /* Only master thread does this */
        if (tid == 0){
            for(i=0; i<nthreads; i++){
                if(strlen(msg_list[i]) < 1){
                    sleep(2);
                }
                strcat(msg2,msg_list[i]);
            }
        }
        
    }  /* All threads join master thread and disband */

    if(my_thread_rank != 0){
        //create message
        dest = 0;
        MPI_Send(msg2, len_msg2 + 1, MPI_CHAR, dest, tag, thread_comm);
    }
    else{  //my rank is 0
        printf("\n\nThreaded Section I/O:\n");
        printf(msg2);
        for(source = 1; source < p/4; source++){
            MPI_Recv(msg2, len_msg2 + 1, MPI_CHAR, source, tag, thread_comm, &status);
            printf(msg2);
        }
        
    }
    MPI_Barrier(thread_comm);
    }
    else{
        // non-thread participants
        MPI_Barrier(thread_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  
// clean up data structures

// destroy groups and communicators
//MPI_Comm_free(&thread_comm);
//MPI_Group_free(&thread_grp);

// Exit threaded section

// Goodbye from processes
  
if(my_rank != 0)
      {
      //create message
      sprintf(message, "Goodbye from process %d (pid %d)!", my_rank, my_pid);
      dest = 0;
      MPI_Send(message, strlen(message) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    }
  else  //my rank is 0
    {
      printf("\n\nGoodbye from process %d (pid %d)!\n", my_rank, my_pid);
      for(source = 1; source < p; source++)
	{
	  MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
	  printf("%s\n", message);
	}
    }
  
  //shutdown MPI
  MPI_Finalize();
  
  return 0;
}
