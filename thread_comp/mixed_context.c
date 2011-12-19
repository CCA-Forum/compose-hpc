/*
 * C version of mixed_context.f but hopefully it works now.
 */

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include "mpi.h"
#include "pblas.h"

extern void Cblacs_get();
extern void Cblacs_pinfo();
extern void Cblacs_exit();
extern void Cblacs_gridexit();
extern void Cblacs_gridinit();
extern void Cblacs_gridinfo();
extern void Cblacs_barrier();
extern int Cblacs_pnum();
extern void descinit_();
extern int numroc_();


void descinit(int * desc, int m, int n, int mb, int nb, int csrc, int rsrc, int icontxt, int lld, int info){

    desc[0] = BLOCK_CYCLIC_2D;
    desc[1] = icontxt;
    desc[2] = m;
    desc[3] = n;
    desc[4] = mb;
    desc[5] = nb;
    desc[6] = csrc;
    desc[7] = rsrc;
    desc[8] = lld;
}

void init_rand_double(double * a, int len){
    int i, j;
    for(i = 0; i < len; i++){
        a[i] = rand() % 4096;
    }
}

void init_zero_double(double * a, int len){
    int i;
    for(i = 0; i < len; i++){
        a[i] = 0.0;
    }
}

void init_zero_int(int * a, int len){
    int i;
    for(i = 0; i < len; i++){
        a[i] = 0;
    }
}

void print_mat_int(FILE * f, int * mat, int dim){
    int i;
    for(i = 0; i < dim; i ++)
        fprintf(f, "%d\n", mat[i]);
}

void print_mat_double(FILE * f, double * mat, int dim){
    int i;
    for(i = 0; i < dim; i ++)
        fprintf(f, "%f\n", mat[i]);
}

int setup_for_threads_smp(int * pmap, int tpn, int icontxt){
    /* Construct the process map pmap and return 1 if the process will move on 
     * to the threaded section, 0 otherwise.
     */
    //printf("in setup for threads\n");
    int npc, npr, mr, mc, myid, id, nproc, i, j;
    int k = 0;
    Cblacs_gridinfo(icontxt, &npr, &npc, &mr, &mc);
    //printf("  %d %d %d %d\n\n", npr, npc, mr, mc);
    myid = Cblacs_pnum(icontxt, mr, mc);
    nproc = npr * npc;
    //pmap = (int *)malloc((nproc / tpn) * sizeof(int));
    for (i = 0; i < npr; i++){
        for (j = 0; j < npc; j++){
            id = Cblacs_pnum(icontxt, i, j);
            if (id % 4 == 0){
                pmap[k] = id;
                k++;
            }
        }
    }
    
    if (myid % 4 == 0){
        //printf("I am proc %d and I get to go into the threaded region\n", myid);
        return 1;
    }
    else
        return 0;
}

void solve_orig(){
    /* The original (process-only) version of the solve
     */

}

int main(int argc, char* argv[]){
    double * p_amat;
    int * p_ipiv;
    double * p_brhs;
    double * work;
    
    double * tp_amat;
    int * tp_ipiv;
    double * tp_brhs;
    double * twork;
    
    // vars for timing
    double t1, dummy, time, second1, tmin;
    double tt1, ttime, ttmin;
    
    // file names
    char suffix[4];
    char amat_infile[50];
    char brhs_infile[50];
    char outfile[50];
    char toutfile[50];
    
    // data for process-only section
    int p_amat_dim, p_brhs_dim, p_ipiv_dim, p_amat_size;
    int irnc, nrow, ncol, lld, nrow_local, ncol_local;
    int mb, nb, myid, nproc, myrow, mycol, nprow, npcol;
    int icontxt, info;
    int desc_amat[9], desc_brhs[9];
    
    // data for threaded section
    int tp_amat_dim, tp_brhs_dim, tp_ipiv_dim, tp_amat_size;
    int tirnc, tnrow, tncol, tlld, tnrow_local, tncol_local;
    int tmb, tnb, tmyid, tnproc, tmyrow, tmycol, tnprow, tnpcol;
    int ticontxt, i, j, n, ni, pi, k;
    int tdesc_amat[9], tdesc_brhs[9];
    
    // vars to keep track of processes and threads
    int num_threads, num_nodes, num_ppn, limit, thread_region, tpn;
    char thread_count[5];
    int * pmap;
    
    // vars for constructing a node and process mapping
    char name[20], her_name[20], tmp_c[20];
    int node_num, her_node_num, name_number, pnum;
    int **node_map;
    int *node_index;
    char fname[50];
    int mypid;
    
    int zero = 0;
    int one = 1;
    int smp = 1;
    
    // this is where we would get our input, but for now we are just gonna hard code some things
    mypid = getpid();
    srand(mypid);
    nprow = 8;
    npcol = 4;
    nb = 2;
    mb = 2;
    num_threads = 4;
    nrow = 16;
    ncol = 16;
    info = 0;
    tpn = 4;
    
    MPI_Init(&argc, &argv);
    // set up context (for processes)
    Cblacs_pinfo(&myid, &nproc);
    Cblacs_get(-1, 0, &icontxt);
    Cblacs_gridinit(&icontxt, COLUMN, nprow, npcol);
    Cblacs_gridinfo(icontxt, &nprow, &npcol, &myrow, &mycol);
    
    if(myrow < nprow && mycol < npcol){
        sprintf(fname, "ofile.%d-%d", myrow, mycol);
        FILE * my_file = fopen(fname, "w");
        // all procs that have a piece of the matrix
        nrow_local = MAX(1, numroc_(&nrow, &mb, &myrow, &zero, &nprow));
        ncol_local = MAX(1, numroc_(&ncol, &nb, &mycol, &zero, &npcol));
        lld = MAX(1, nrow_local);
        
        p_amat_dim = nrow_local * ncol_local;
        p_amat_size = p_amat_dim * 16 / 1.0e+06;
        
        p_amat = (double *)malloc(p_amat_dim * sizeof(double));
        init_rand_double(p_amat, p_amat_dim);
        work = (double *)malloc(p_amat_dim * sizeof(double));
        init_zero_double(work, p_amat_dim);
        
        p_brhs_dim = nrow;
        p_ipiv_dim = nrow;
        p_brhs = (double *)malloc(p_brhs_dim * sizeof(double));
        p_ipiv = (int *)malloc(p_ipiv_dim * sizeof(int));
        init_zero_double(p_brhs, p_brhs_dim);
        init_zero_int(p_ipiv, p_ipiv_dim);
        
        descinit(desc_amat, nrow, ncol, mb, nb, zero, zero,	\
                 icontxt, lld, info);
        if (info != 0)
            printf("problem constructing desc_amat in proc %d, %d", myrow, mycol);
        
        descinit(desc_brhs, nrow, one, mb, nb, zero, zero,	\
                 icontxt, lld, info);
        if (info != 0)
            printf("problem constructing desc_brhs in proc %d, %d", myrow, mycol);
        
        
        fprintf(my_file, "My p_amat:\n");
        print_mat_double(my_file, p_amat, p_amat_dim);
        fprintf(my_file, "My p_brhs:\n");
        print_mat_double(my_file, p_brhs, p_brhs_dim);
        fprintf(my_file, "My p_ipiv:\n");
        print_mat_int(my_file, p_ipiv, p_ipiv_dim);
        
        if (myid == 0)
            printf("Initialized Matrices\n");
        
        Cblacs_barrier(icontxt, ALL);
        
        pdgetrf_(&nrow, &ncol, p_amat, &one, &one, desc_amat, p_ipiv, &info);
        if(info != 0){
            printf("pdgetrf returns info = %d\n", info);
        }
        else{
            Cblacs_barrier(icontxt, ALL);
            pdgetrs_(NO, &nrow, &one, p_amat, &one, &one, desc_amat, p_ipiv, p_brhs, &one, &one, desc_brhs, &info);
            if(info != 0){
                printf("pdgetrs returns info = %d\n", info);
            }
        }
        
        fprintf(my_file, "My answer is:\n");
        for (i=0; i < p_brhs_dim; i++){
            fprintf(my_file, "%f\n", p_brhs[i]);
        }
        
        if (myid == 0)
            printf("Success!\n");
        
        Cblacs_barrier(icontxt, ALL);
        
        // Do the same thing, only using threads and a new context
        if (smp == 1){
            pmap = (int *)malloc((nproc / tpn) * sizeof(int));
            thread_region = setup_for_threads_smp(pmap, tpn, icontxt);
        }
        else {
            thread_region = setup_for_threads(pmap, tpn, icontxt); 
        }
        tnproc = nproc / tpn;
        tnprow = nprow / 2;
        tnpcol = npcol / 2;
        
        if (myid == 0){
            for (i = 0; i < tnproc; i++)
                printf("%d, ", pmap[i]);
        }
        // threaded region
        // create new context using pmap
        Cblacs_get(icontxt, 10, &ticontxt); // populate ticontxt with value of icontxt
        Cblacs_gridmap(&ticontxt, pmap, tnprow, tnprow, tnpcol);
        Cblacs_gridinfo(ticontxt, &tnprow, &tnpcol, &tmyrow, &tmycol);

        if (tmyrow < tnprow && tmycol < tnpcol) {
            tmyid = Cblacs_pnum(ticontxt, tmyrow, tmycol);
            
            if (tmyid == 0)
                printf("\n-----------------------------\nEnter Threaded Region\n");
            // generate matrix    
            if(tmyrow < tnprow && tmycol < tnpcol){
                // all procs that have a piece of the matrix     
                tnrow_local = MAX(1, numroc_(&nrow, &mb, &tmyrow, &zero, &tnprow));
                tncol_local = MAX(1, numroc_(&ncol, &nb, &tmycol, &zero, &tnpcol));
                tlld = MAX(1, tnrow_local);
                
                tp_amat_dim = tnrow_local * tncol_local;
                tp_amat_size = tp_amat_dim * 16 / 1.0e+06;
                
                if(tmyid == 0){
                    printf("nrow = %d\n", nrow);
                    printf("ncol = %d\n", ncol);
                    printf("mb = %d\n", mb);
                    printf("nb = %d\n", nb);
                    printf("tnprow = %d\n", tnprow);
                    printf("tnpcol = %d\n", tnpcol);
                    printf("tnrow_local = %d\n", tnrow_local);
                    printf("tncol_local = %d\n", tncol_local);
                    printf("tp_amat_dim = %d\n", tp_amat_dim);
                }
                
                tp_amat = (double *)malloc(tp_amat_dim * sizeof(double));
                init_rand_double(tp_amat, tp_amat_dim);
                twork = (double *)malloc(tp_amat_dim * sizeof(double));
                init_zero_double(twork, tp_amat_dim);
                
                tp_brhs_dim = nrow;
                tp_ipiv_dim = nrow;
                tp_brhs = (double *)malloc(tp_brhs_dim * sizeof(double));
                tp_ipiv = (int *)malloc(tp_ipiv_dim * sizeof(int));
                init_zero_double(tp_brhs, tp_brhs_dim);
                init_zero_int(tp_ipiv, tp_ipiv_dim);
                
                printf("attempting to make desc's (%d)\n", mypid);
                
                descinit(tdesc_amat, nrow, ncol, mb, nb, zero, zero,	\
                         ticontxt, tlld, info);
                if (info != 0)
                    printf("problem constructing desc_amat in proc %d, %d", myrow, mycol);
                else
                    print_mat_int(my_file, tdesc_amat, 9);
                
                descinit(tdesc_brhs, nrow, one, mb, nb, zero, zero,	\
                         ticontxt, tlld, info);
                if (info != 0)
                    printf("problem constructing desc_brhs in proc %d, %d", tmyrow, tmycol);
                else
                    print_mat_int(my_file, tdesc_amat, 9);
                
                fprintf(my_file, "My tp_amat:\n");
                print_mat_double(my_file, tp_amat, tp_amat_dim);
                fprintf(my_file, "My tp_brhs:\n");
                print_mat_double(my_file, tp_brhs, tp_brhs_dim);
                fprintf(my_file, "My tp_ipiv:\n");
                print_mat_int(my_file, tp_ipiv, tp_ipiv_dim);
                
                if (tmyid == 0)
                    printf("Initialized Matrices\n");
                
                Cblacs_barrier(ticontxt, ALL);
                
                // turn on threads
                omp_set_num_threads(tpn);
                
                // solve matrix
                pdgetrf_(&nrow, &ncol, tp_amat, &one, &one, tdesc_amat, tp_ipiv, &info);
                if(info != 0){
                    printf("pdgetrf returns info = %d\n", info);
                }
                else{
                    Cblacs_barrier(ticontxt, ALL);
                    pdgetrs_(NO, &nrow, &one, tp_amat, &one, &one, tdesc_amat, tp_ipiv, tp_brhs, &one, &one, tdesc_brhs, &info);
                    if(info != 0){
                        printf("pdgetrs returns info = %d\n", info);
                    }
                }
                Cblacs_barrier(ticontxt, ALL);
                
                fprintf(my_file, "My answer is:\n");
                for (i=0; i < tp_brhs_dim; i++){
                    fprintf(my_file, "%f\n", tp_brhs[i]);
                }
                if(tmyid == 0){
                    printf("Success!\n");
                }
                
                // turn off threads
                omp_set_num_threads(tpn);
                
                // print stats
                if (tmyid == 0){
                    printf("\n-----------------------------\nExit threaded section\n");
                }
            }
            // destroy contxt
            
            // free data
            free(tp_amat);
            free(twork);
            free(tp_brhs);
            free(tp_ipiv);
            free(pmap);
            Cblacs_gridexit(ticontxt);
        }
        Cblacs_barrier(icontxt, ALL);
        // close files
        fclose(my_file);
        
        // free data
        free(p_amat);
        free(work);
        free(p_brhs);
        free(p_ipiv);
        Cblacs_gridexit(icontxt);    
    }
    // done!
    
    Cblacs_exit(0);
    return 0;
    
}
