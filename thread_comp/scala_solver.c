/*
 * C version of mixed_context.f but hopefully it works now.
 */

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
//#include <complex.h>
#include <sys/types.h>
#include "mpi.h"
#include "pblas.h"
//#include "Bconfig.h"
//#include "Bdef.h"
//#include "pxsyevx.h" // not sure if this is needed
//#include "tools.h"   // not sure if this is needed

extern void Cblacs_get();
extern void Cblacs_pinfo();
extern void Cblacs_exit();
extern void Cblacs_gridexit();
extern void Cblacs_gridinit();
extern void Cblacs_gridinfo();
extern void Cblacs_barrier();
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

/*
void init_rand_double(double * a, int len){
  int i;
  for(i = 0; i < len; i++){
    a[i] = rand() / 3.0;
  }
}
*/

void print_mat_int(FILE * f, int * mat, int dim){
  int i;
  for(i = 0; i < dim; i ++)
    fprintf(f, "%d\n", mat[i]);
}

void print_mat_double(FILE * f, double * mat, int dim){
  int i;
  printf("in print_mat_double");
  for(i = 0; i < dim; i ++)
    fprintf(f, "%f\n", mat[i]);
}



int main(int argc, char* argv[]){
  double * p_amat;
  int * p_ipiv;
  double * p_brhs;
  double * work;

  // data for process-only section
  int p_amat_dim, p_brhs_dim, p_ipiv_dim, p_amat_size;
  int nrow, ncol, lld, nrow_local, ncol_local;
  int mb, nb, myid, nproc, myrow, mycol, nprow, npcol;
  int icontxt, info;
  int desc_amat[9], desc_brhs[9];

  int i, j, n, k;
  
  char fname[50];

  int zero = 0;
  int one = 1;

  // this is where we would get our input, but for now we are just gonna hard code some things

  srand(getpid());
  nprow = 2;
  npcol = 2;
  nb = 4;
  mb = 4;
  nrow = 16;
  ncol = 16;
  info = 0;
  
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
    printf("\n\nIn the branch, after max\n\n");
    
    p_amat_dim = nrow_local * ncol_local;
    p_amat_size = p_amat_dim * 16 / 1.0e+06;

    printf("nrow = %d\n", nrow);
    printf("ncol = %d\n", ncol);
    printf("mb = %d\n", mb);
    printf("nb = %d\n", nb);
    printf("nprow = %d\n", nprow);
    printf("npcol = %d\n", npcol);
    printf("nrow_local = %d\n", nrow_local);
    printf("ncol_local = %d\n", ncol_local);
    printf("p_amat_dim = %d\n", p_amat_dim);

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
    else
      print_mat_int(my_file, desc_amat, 9);
    descinit(desc_brhs, nrow, one, mb, nb, zero, zero,	\
	      icontxt, lld, info);
    if (info != 0)
      printf("problem constructing desc_brhs in proc %d, %d", myrow, mycol);
    else
      print_mat_int(my_file, desc_amat, 9);

    for(i=0;i<9;i++)
      printf("%d, ", desc_amat[i]);
    printf("\n");

    for(i=0;i<9;i++)
      printf("%d, ", desc_brhs[i]);
    printf("\n");

    fprintf(my_file, "My p_amat:\n");
    print_mat_double(my_file, p_amat, p_amat_dim);
    fprintf(my_file, "My p_brhs:\n");
    print_mat_double(my_file, p_brhs, p_brhs_dim);
    fprintf(my_file, "My p_ipiv:\n");
    print_mat_int(my_file, p_ipiv, p_ipiv_dim);

    
    printf("Initialized Matrices\n");

    Cblacs_barrier(icontxt, ALL);
    
    pdgetrf_(&nrow, &ncol, p_amat, &one, &one, desc_amat, p_ipiv, &info);
    if(info != 0){
      printf("pdgetrf returns info = %d\n", info);
    }
    else{
      pdgetrs_(NO, &nrow, &one, p_amat, &one, &one, desc_amat, p_ipiv, p_brhs, &one, &one, desc_brhs, &info);
      if(info != 0){
	printf("pdgetrs returns info = %d\n", info);
      }
    }
    
    fprintf(my_file, "My answer is:\n");
    for (i=0; i < p_brhs_dim; i++){
      fprintf(my_file, "%f\n", p_brhs[i]);
    }
    printf("Success!\n");

  }
  

  // done!
  Cblacs_gridexit(icontxt);
  Cblacs_exit(0);
  return 0;

}
