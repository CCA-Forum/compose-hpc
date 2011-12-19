#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "mpi.h"

#define SUCCESS 1
#define FAILURE 0
#define ALL 'A'
#define EACH 'E'
#define ONE 'O'

int deep_copy(int * src, int ** dest, int len){
    int i;
    //printf("in deep_copy...");
    // this can probably be parallelized
    for(i=0;i<len;i++)
        (*dest)[i] = src[i];
    //printf("success!\n");
}

int method_wrapper(int * in, int len_in, int ** out, int * len_out, 
                   char call_type, char return_type, int nthreads, 
                   MPI_Comm caller_comm){
    /* wrapper for threaded method.  return status of call.
     * *call_type* refers to the data distribution of the caller(s)
     * *return_type* refers to the data distribution of the receivers(s)
     * (everyone always calls and receives, but may not always get data)
     */
    int status;
    MPI_Status ecode;
    int myrank, rc;
    int nprocs;
    int * tin;
    int * tout;
    int * lens;
    int has_data, enter_thread_reg;
    int tlen_in, tlen_out, tmyrank, nmyrank, tsize, nsize;
    int i, send_data_to, curr_pos;
    MPI_Comm thread_comm, node_cohort;
    MPI_Comm_rank(caller_comm, &myrank);
    MPI_Comm_size(caller_comm, &nprocs);

    MPI_Barrier(caller_comm);
    // elect processes to go into threaded section
    if(myrank % nthreads == 0){
        enter_thread_reg = 1;
        send_data_to = nthreads * (myrank/nthreads);
    }
    else{
        enter_thread_reg = 0;
        send_data_to = nthreads * (myrank/nthreads);
    }
    
    MPI_Comm_split(caller_comm, enter_thread_reg, myrank, &thread_comm);
    MPI_Comm_rank(thread_comm, &tmyrank);
	MPI_Comm_size(thread_comm, &tsize);
	MPI_Comm_split(caller_comm, send_data_to, myrank, &node_cohort);
	MPI_Comm_rank(node_cohort, &nmyrank);
	MPI_Comm_size(node_cohort, &nsize);

	tout = NULL;
    // move data from processes that are not participating to those that are
    // * need to think about how to keep data where it is as much as possible
    /*** ALL ***/
    if(call_type == ALL){
        if(enter_thread_reg == 1){
            // need to gather all the data from the other processes
            //tlen_in = len_in;
			MPI_Reduce(&len_in, &tlen_in, 1, MPI_INT, MPI_SUM, 0, node_cohort);
			tin = (int *) malloc(tlen_in * sizeof(int));
			MPI_Gather(in, len_in, MPI_INT, tin, len_in, MPI_INT, 0, node_cohort);
        }
        else{
			MPI_Reduce(&len_in, &tlen_in, 1, MPI_INT, MPI_SUM, 0, node_cohort);
			MPI_Gather(in, len_in, MPI_INT, tin, len_in, MPI_INT, 0, node_cohort);
        }
    }
    /*** EACH ***/
    else if(call_type == EACH){
        if(enter_thread_reg == 1){
            // need to gather all the data from the other processes
            //tlen_in = len_in;
			MPI_Reduce(&len_in, &tlen_in, 1, MPI_INT, MPI_SUM, 0, node_cohort);
			tin = (int *) malloc(tlen_in * sizeof(int));
			MPI_Gather(in, len_in, MPI_INT, tin, len_in, MPI_INT, 0, node_cohort);
        }
        else{
			MPI_Reduce(&len_in, &tlen_in, 1, MPI_INT, MPI_SUM, 0, node_cohort);
			MPI_Gather(in, len_in, MPI_INT, tin, len_in, MPI_INT, 0, node_cohort);
        }
    }
    /*** ONE ***/
    else{
        if(enter_thread_reg == 1 && myrank == 0){
			tin = in;
			tlen_in = len_in;
        }
		else {
			tin = NULL;
			tlen_in = 0;
		}

    }

    /*** START THREADED REGION ***/
    status = FAILURE;
    omp_set_num_threads(nthreads);
    if(enter_thread_reg == 1){
        // call threaded_method
        // ** do i need to provide the communicator too???
        status = threaded_method(tin, tlen_in, &tout, &tlen_out, thread_comm);
        // barrier
        MPI_Barrier(thread_comm);
    }
    omp_set_num_threads(1);
    /*** END THREADED REGION ***/

    // broadcast status message
    MPI_Bcast(&status, 1, MPI_INT, 0, caller_comm);

    /*** return_type == ALL ***/
    if(return_type == ALL){
        if(enter_thread_reg == 1){
			(*len_out) = tlen_out;
            (*out) = (int *) malloc(*len_out * sizeof(int));
            deep_copy(tout, out, *len_out);
	
			MPI_Bcast(len_out, 1, MPI_INT, 0, node_cohort);
			MPI_Bcast((*out), (*len_out), MPI_INT, 0, node_cohort);
            
        }
        else{
			MPI_Bcast(len_out, 1, MPI_INT, 0, node_cohort);
            if((*len_out) > 1)
                (*out) = (int *)malloc((*len_out) * sizeof(int));
			MPI_Bcast((*out), (*len_out), MPI_INT, 0, node_cohort);
        }
    }
    /*** return_type == EACH ***/
    else if(return_type == EACH){
        if(enter_thread_reg == 1){
            (*len_out) = tlen_out / nthreads;
			(*out) = (int *) malloc((*len_out) * sizeof(int));
			MPI_Bcast(len_out, 1, MPI_INT, 0, node_cohort);
			// problem in the following scatter....
			MPI_Scatter(tout, (*len_out), MPI_INT, (*out), (*len_out), MPI_INT, 0, node_cohort);
        }
        else{
			MPI_Bcast(len_out, 1, MPI_INT, 0, node_cohort);
			if((*len_out) > 1)
				(*out) = (int *) malloc((*len_out) * sizeof(int));
			MPI_Scatter(tout, (*len_out), MPI_INT, (*out), (*len_out), MPI_INT, 0, node_cohort);
        }
    }
    /*** return_type == ONE ***/
    else{
        // whatever data is in the process that originally passed the data in
		if (enter_thread_reg == 1) {
			// gather all data from all threaded procs to myrank == 0
			MPI_Reduce(&tlen_out, len_out, 1, MPI_INT, MPI_SUM, 0, thread_comm);
			if (tmyrank == 0){
				(*out) = (int *) malloc((*len_out) * sizeof(int));
			}
			else{
				(*len_out) = 0;
			}
			MPI_Gather(tout, tlen_out, MPI_INT, (*out), tlen_out, MPI_INT, 0, thread_comm);
		}
		else {
			(*len_out) = 0;
		}
    }

	if(tout != NULL){
		free(tout);
	}
    MPI_Barrier(caller_comm);
    return status;
}

int threaded_method(int * in, int len_in, int ** out, int * len_out, MPI_Comm my_comm){
    /* threaded method.  return status of call.
     */
    int i;
    int myrank, p;
    int tid, nthreads, len;
    //int * mydata;
    //int * sum;
    int chunk_start;
    MPI_Comm_rank(my_comm, &myrank);
    for(i=0;i<len_in;i++){
        printf("   in[%d] = %d (%d)\n", i, in[i], myrank);
    }

#pragma omp parallel shared(out) private(tid, chunk_start, i)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        len = nthreads;
        p = len_in / nthreads;
        #pragma omp single
        {
            (*out) = (int *) malloc(nthreads * sizeof(int));
            //printf("\n\n\nlen_in %d, len_out %d, nthreads %d, p %d\n\n\n", len_in, len, nthreads, p);
        }
        #pragma omp barrier

        chunk_start = tid * nthreads;
        (*out)[tid] = 0;
        for(i=0;i<nthreads;i++){
			if (len_in > 0)
				(*out)[tid] += in[i + chunk_start];
			else
				(*out)[tid] = 0;
        }
    }
    for(i=0;i<len;i++)
        printf("       out[%d] = %d (%d)\n", i, (*out)[i], myrank);
	*len_out = len;
    return SUCCESS;
}


int main(int argc, char* argv[])
{
  int my_rank, my_thread_rank;
  int p, rc;
  int source;
  int dest;
  int tag = 0;
  char message[100];
  char hname[50];
  pid_t my_pid;
  MPI_Status status;

  int * in;
  int * out;
  int len_in, len_out;
  char call_type, return_type;
  int retval;

  // to merge threaded and non-threaded sections
  MPI_Group thread_grp, process_grp;
  MPI_Comm thread_comm;
  int * grp_list;
  int enter_thread_reg;
  int len_grp_list;
  int i;

  // for threaded stuff
  int nthreads = 4;
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
      printf("\n--------------------\n\n");
  }
  /*****************************************************************/
  MPI_Barrier(MPI_COMM_WORLD);

  // set up input data
  len_in = p * nthreads;
  in = (int *) malloc(len_in * sizeof(int));
  for(i=0; i<len_in; i++) in[i] = my_rank + i;
  //out = NULL;
  len_out = 1;
  //out = (int *) malloc(len_out * sizeof(int));
  call_type = ALL;
  //call_type = EACH;
  //call_type = ONE;
  //return_type = ALL;
  //return_type = EACH;
  return_type = ONE;	
	
  /*********************  CALL METHOD  *****************************/
  retval = method_wrapper(in, len_in, &out, &len_out, call_type, return_type, 
                          nthreads, MPI_COMM_WORLD);
  /*****************************************************************/
  if (return_type == ALL || return_type == EACH){
	  if(my_rank == 0){
		  if(retval == FAILURE)
			  printf("failure reported from process %d\n", my_rank);
		  else{
			  printf("results from process %d: ", my_rank);
			  for(i=0; i<len_out; i++){
				  printf("   %d\n", out[i]);
			  }
			  printf("\n");
		  }
		  for(source = 1; source < p; source++){
			  MPI_Recv(&retval, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
			  if(retval == FAILURE) 
				  printf("failure reported from process %d\n", my_rank);
			  else{
				  if (return_type == ALL || return_type == EACH){
					  MPI_Recv(&len_out, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
					  MPI_Recv(out, len_out, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
					  printf("results from process %d: ", source);
					  for(i=0; i<len_out; i++){
						  printf("   %d\n", out[i]);
					  }
					  printf("\n");
				  }
			  }
		  }
		  printf("Done presenting results from call\n");
	  }
	  else{
		  MPI_Send(&retval, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		  if(retval == SUCCESS){
			  MPI_Send(&len_out, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
			  MPI_Send(out, len_out, MPI_INT, 0, tag, MPI_COMM_WORLD);
		  }
	  }
  }
  else{
	if (my_rank == 0) {
		if (retval == FAILURE) {
			printf("failure reported from process %d\n", my_rank);
		}
		else{
			printf("results from process %d: ", my_rank);
			for(i=0; i<len_out; i++)
				printf("   %d\n", out[i]);
			printf("\n");
			printf("Done presenting results from call\n");
		}
	}	
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // clean up data structures
  free(in);
  if(len_out > 1)
	free(out);
  // shutdown MPI
  MPI_Finalize();
  
  return 0;
}
