/* Allocation Topography Discovery Tool
 *
 * This utility discovers the topology for the allocation
 * and writes it to a file.
 *
 * Based on the example program for using hwloc.
 * Copyright © 2009-2010 INRIA
 * Copyright © 2009-2010 Université Bordeaux 1
 * Copyright © 2009-2010 Cisco Systems, Inc.  All rights reserved.
 *
 * hwloc-hello.c
 */

#include <hwloc.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "mpi.h"



int main(int argc, char* argv[])
{

    char hname[50];
    int topo_depth, hnlen, depth;
    hwloc_topology_t topology;
    hwloc_cpuset_t cpuset;
    hwloc_obj_t obj;
    int my_pu = -1;
    pid_t my_pid;

    int my_rank;
    int p, i;
    int source;
    int dest;
    int tag = 0;
    char message[100];
    char string[100];
    MPI_Status status;

    int num, next, from;

    //Start up MPI                                                                                                                         
    MPI_Init(&argc, &argv);

    //find process rank                                                                                                                    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //find number of procs                                                                                                                 
    MPI_Comm_size(MPI_COMM_WORLD, &p);



    // all procs get hostname, topo and PU #
    gethostname(hname, 100);
    my_pid = getpid();

    /* Allocate and initialize topology object. */
    hwloc_topology_init(&topology);

    /* Perform the topology detection. */
    hwloc_topology_load(topology);
    topo_depth = hwloc_topology_get_depth(topology);

    /* get the core that this process is bound to */

    cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);
    
    my_pu = hwloc_bitmap_first(cpuset);

    hwloc_bitmap_free(cpuset);

    // if rank 0, print stuff, else, send stuff
    if(my_rank != 0)
        {
            //create message with rank, PU # and hostname     
            sprintf(message, "Greetings from process %d running on PU # %d on %s!", my_rank, my_pu, hname);
            dest = 0;
            MPI_Send(message, strlen(message) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        }
    else  //my rank is 0
        {
            // print local topology
            /*for (depth = 0; depth < topo_depth; depth++) {
                printf("*** Objects at level %d\n", depth);
                for (i = 0; i < hwloc_get_nbobjs_by_depth(topology, depth); i++) {
                    hwloc_obj_snprintf(string, sizeof(string), topology, hwloc_get_obj_by_depth(topology, depth, i), "#", 0);
                    printf("Index %u: %s\n", i, string);
                }
                }*/

            printf("\n\nGreetings from process %d running on PU # %d on %s!\n", my_rank, my_pu, hname);
            // get hello messages with rank, PU #, and host name
            for(source = 1; source < p; source++)
                {
                    MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
                    printf("%s\n", message);
                }
        }

    sleep(5);
    /*
    next = (my_rank + p + 1) % p;
    from = (my_rank + p - 1) % p;
    num = 100;

    if (my_rank == 0) {
        printf("Process %d sending %d to %d\n", my_rank, num, next);
        MPI_Send(&num, 1, MPI_INT, next, tag, MPI_COMM_WORLD); 
    }
    */
    /* Pass the message around the ring.  The exit mechanism works */
    /* as follows: the message (a positive integer) is passed */
    /* around the ring.  Each time is passes rank 0, it is decremented. */
    /* When each processes receives the 0 message, it passes it on */
    /* to the next process and then quits.  By passing the 0 first, */
    /* every process gets the 0 message and can quit normally. */
    /*
    while (1) {

        MPI_Recv(&num, 1, MPI_INT, from, tag, MPI_COMM_WORLD, &status);
        //printf("Process %d received %d\n", rank, num);

        if (my_rank == 0) {
            num--;
            //printf("Process 0 decremented num\n");
        }

        //printf("Process %d sending %d to %d\n", rank, num, next);
        MPI_Send(&num, 1, MPI_INT, next, tag, MPI_COMM_WORLD);

        if (num == 0) {
            //printf("Process %d exiting\n", my_rank);
            break;
        }
    }
    */
    /* The last process does one extra send to process 0, which needs */
    /* to be received before the program can exit */
    /*
    if (my_rank == 0)
        MPI_Recv(&num, 1, MPI_INT, from, tag, MPI_COMM_WORLD, &status);
    */

    /* Quit */

    hwloc_topology_destroy(topology);

    //shutdown MPI
    MPI_Finalize();
    return 0;
}
