#include "cublas.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include "cblas.h"
extern void tce_sort_2_(double *,double *,int *,int *,int *,int *,double *);
extern void tce_sort_4_(double *,double *,int *,int *,int *,int *,int *,int *,int *,int *,double *);
#ifdef __cplusplus
}
#endif /* __cplusplus */
typedef unsigned long long hrtime_t;

inline hrtime_t gethrtime()
{
  union __unnamed_class___F0_L23_C5_L50R__L51R__scope____SgSS2___variable_declaration__variable_type__variable_name_L50R__L51R__scope____SgSS2____scope_____DELIMITER__L50R__L51R__scope____SgSS2___variable_declaration__variable_type_hrtime_tUL__typedef_declaration_variable_name_L50R__L51R__scope____SgSS2____scope__ll {
  struct __unnamed_class___F0_L26_C9_L52R_variable_declaration__variable_type_Ui_variable_name_L52R__scope__a__DELIMITER__L52R_variable_declaration__variable_type_Ui_variable_name_L52R__scope__d {
  unsigned int a;
  unsigned int d;}_;
  hrtime_t ll;}_;
  asm volatile ("rdtsc" : "=a" (_._.a), "=d" (_._.d));
  return (_.ll / 2.394);
}

int main(int argc,char *argv[])
{
  FILE *runtime;
  FILE *dgemm1time;
  FILE *dgemm2time;
  FILE *dgemm3time;
  FILE *dgemm4time;
  FILE *trans1time;
  FILE *trans2time;
  FILE *inittime;
  FILE *totaldgemmtime;
  FILE *totaltranstime;
  int at;
  int bt;
  int ct;
  int dt;
  int a;
  int b;
  int c;
  int d;
  int V = atoi(argv[1]);
  int O = atoi(argv[2]);
  int pt;
  int qt;
  int rt;
  int st;
  int p;
  int q;
  int r;
  int s;
  int i;
  int j;
  int k;
  int l;
  long ns;
  long nss;
  int O2 = (O * O);
  int O3 = (O2 * O);
  double dif;
  double factor;
  double rd;
  hrtime_t dgemm_strt;
  hrtime_t dgemm_end;
  hrtime_t trans_strt;
  hrtime_t trans_end;
  hrtime_t array_init_strt;
  hrtime_t array_init_end;
  hrtime_t copy_strt;
  hrtime_t copy_end;
  hrtime_t prog_strt;
  hrtime_t prog_end;
  double array_init;
  double total_dgemm_time;
  double total_transpose_time;
  double total_copy;
  double copy1;
  double copy2;
  double copy3;
  double dgemm1;
  double dgemm2;
  double dgemm3;
  double dgemm4;
  double trans1;
  double trans2;
  double trans3;
  double transB;
  double prog;
  double *B1 = (double *)(malloc(((((V * V) * V) * V) * (sizeof(double )))));
  double *C1 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C2 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C3 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C4 = (double *)(malloc(((O * V) * (sizeof(double )))));
//double *C1_T = (double *)malloc(O*V*sizeof(double));
  double *C2_T = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C3_T = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C4_T = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *A1 = (double *)(malloc(((((O * O) * O) * O) * (sizeof(double )))));
  double *A1_T = (double *)(malloc(((((O * O) * O) * O) * (sizeof(double )))));
  double *t1 = (double *)(malloc(((O * O) * (sizeof(double )))));
  double *t2 = (double *)(malloc((O * (sizeof(double )))));
  double t3 = 0;
//printf("V=%d, O=%d\n",V,O);
  ((prog_strt = 0) , (array_init_strt = 0));
  ((((((dgemm1 = 0) , (dgemm2 = 0))) , (dgemm3 = 0))) , (dgemm4 = 0));
  ((((((trans1 = 0) , (trans2 = 0))) , (trans3 = 0))) , (transB = 0));
  ((((copy1 = 0) , (copy2 = 0))) , (copy3 = 0));
  prog_strt = gethrtime();
  array_init_strt = gethrtime();
  for (s = 0; s < O; s++) {
    for (d = 0; d < V; d++) {
      rd = (rand() * 10);
      C1[(s * V) + d] = rd;
      C2[(s * V) + d] = rd;
      C3[(s * V) + d] = rd;
      C4[(s * V) + d] = rd;
    }
  }
  for (p = 0; p < O; p++) {
    for (q = 0; q < O; q++) {
      for (r = 0; r < O; r++) {
        for (s = 0; s < O; s++) {
          A1[(((((p * O) + q) * O) + r) * O) + s] = (rand() * 10);
        }
      }
    }
  }
  array_init_end = gethrtime();
  array_init = (array_init_end - array_init_strt);
  trans_strt = gethrtime();
  ((((i = 2) , (j = 1))) , (factor = 1.0));
  tce_sort_2_((C4 + 0),(C4_T + 0),&O,&V,&i,&j,&factor);
  tce_sort_2_((C3 + 0),(C3_T + 0),&O,&V,&i,&j,&factor);
  tce_sort_2_((C2 + 0),(C2_T + 0),&O,&V,&i,&j,&factor);
//tce_sort_2_(&C1[0],&C1_T[0],&O,&V,&i,&j,&factor);
//free(C1);
  free(C2);
  free(C3);
  free(C4);
  trans_end = gethrtime();
  trans1 = (trans_end - trans_strt);
  trans_strt = gethrtime();
  factor = 1.0;
  i = 4;
  j = 3;
  k = 2;
  l = 1;
  tce_sort_4_((A1 + 0),(A1_T + 0),&O,&O,&O,&O,&i,&j,&k,&l,&factor);
  free(A1);
  trans_end = gethrtime();
  trans2 = (trans_end - trans_strt);
  for (a = 0; a < V; a++) {
    for (s = 0; s < O; s++) {
      /*% BLAS_TO_CUBLAS prefix=device1 */
      
double *device1_X;
int sizeType_device1 = sizeof(double);

/* Allocate device memory */
cublasAlloc(O2, sizeType_device1, (void **)&device1_X);

/* Copy matrix, vectors to device */
cublasSetVector(O2, sizeType_device1, t1, 1, device1_X, 1);

/* CUBLAS call */
cublasDscal(O2, 0.0, device1_X, 1);

/* Copy result vector back to host */
cublasSetVector(O2, sizeType_device1, device1_X, 1, t1, 1);

/* Free device memory */
cublasFree(device1_X);

      dgemm_strt = gethrtime();
      /*% BLAS_TO_CUBLAS prefix=device3 lenX=O lenY=O */
      
double *device3_A;
double *device3_X;
double *device3_Y;
int sizeType_device3 = sizeof(double);

/* Allocate device memory */
cublasAlloc(O2 * O, sizeType_device3, (void **)&device3_A);
cublasAlloc(O, sizeType_device3, (void **)&device3_X);
cublasAlloc(O, sizeType_device3, (void **)&device3_Y);

/* Copy matrix, vectors to device */
cublasSetMatrix(O2, O, sizeType_device3, (void *)(A1_T + (s * O3)), O2,
                (void *)device3_A, O2);
cublasSetVector(O, sizeType_device3, (C4_T + (a * O)), 1, device3_X, 1);
if (1.0 != 0)
  cublasSetVector(O, sizeType_device3, t1, 1, device3_Y, 1);

  /* CUBLAS call */
  //BLAS_TO_CUBLAS transformation performance warning: 
  //CUBLAS calls assume arrays are stored in column-major order. 
  //The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemv('N', O2, O, 1.0, device3_A, O, device3_X, 1, 1.0, device3_Y, 1);

/* Copy result vector back to host */
cublasSetVector(O, sizeType_device3, device3_Y, 1, t1, 1);

/* Free device memory */
cublasFree(device3_A);
cublasFree(device3_X);
cublasFree(device3_Y);

      dgemm_end = gethrtime();
      dgemm1 += (dgemm_end - dgemm_strt);
      for (b = 0; b < V; b++) {
        /*% BLAS_TO_CUBLAS prefix=device2 */
        
double *device2_X;
int sizeType_device2 = sizeof(double);

/* Allocate device memory */
cublasAlloc(O, sizeType_device2, (void **)&device2_X);

/* Copy matrix, vectors to device */
cublasSetVector(O, sizeType_device2, t2, 1, device2_X, 1);

/* CUBLAS call */
cublasDscal(O, 0.0, device2_X, 1);

/* Copy result vector back to host */
cublasSetVector(O, sizeType_device2, device2_X, 1, t2, 1);

/* Free device memory */
cublasFree(device2_X);


        dgemm_strt = gethrtime();
        /*% BLAS_TO_CUBLAS prefix=device4 lenX=O lenY=O */
        
double *device4_A;
double *device4_X;
double *device4_Y;
int sizeType_device4 = sizeof(double);

/* Allocate device memory */
cublasAlloc(O * O, sizeType_device4, (void **)&device4_A);
cublasAlloc(O, sizeType_device4, (void **)&device4_X);
cublasAlloc(O, sizeType_device4, (void **)&device4_Y);

/* Copy matrix, vectors to device */
cublasSetMatrix(O, O, sizeType_device4, (void *)t1, O, (void *)device4_A, O);
cublasSetVector(O, sizeType_device4, (C3_T + (b * O)), 1, device4_X, 1);
if (1.0 != 0)
  cublasSetVector(O, sizeType_device4, t2, 1, device4_Y, 1);

  /* CUBLAS call */
  //BLAS_TO_CUBLAS transformation performance warning: 
  //CUBLAS calls assume arrays are stored in column-major order. 
  //The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemv('N', O, O, 1.0, device4_A, O, device4_X, 1, 1.0, device4_Y, 1);

/* Copy result vector back to host */
cublasSetVector(O, sizeType_device4, device4_Y, 1, t2, 1);

/* Free device memory */
cublasFree(device4_A);
cublasFree(device4_X);
cublasFree(device4_Y);

        dgemm_end = gethrtime();
        dgemm2 += (dgemm_end - dgemm_strt);
        for (c = 0; c < V; c++) {
          t3 = 0;
          dgemm_strt = gethrtime();
          t3 = cblas_ddot(O,(C2_T + (c * O)),1,t2,1);
          dgemm_end = gethrtime();
          dgemm3 += (dgemm_end - dgemm_strt);
          dgemm_strt = gethrtime();
          /*% BLAS_TO_CUBLAS prefix=device5 */
          
double *device5_X;
double *device5_Y;
int sizeType_device5 = sizeof(double);

/* Allocate device memory */
cublasAlloc(V, sizeType_device5, (void **)&device5_X);
cublasAlloc(V, sizeType_device5, (void **)&device5_Y);

/* Copy matrix, vectors to device */
cublasSetVector(V, sizeType_device5, (C1 + (s * V)), 1, device5_X, 1);
cublasSetVector(V, sizeType_device5, (B1 + (((((a * V) + b) * V) + c) * V)),
                1, device5_Y, 1);

/* CUBLAS call */
cublasDaxpy(V, t3, device5_X, 1, device5_Y, 1);

/* Copy result vector back to host */
cublasSetVector(V, sizeType_device5, device5_Y, 1,
                (B1 + (((((a * V) + b) * V) + c) * V)), 1);

/* Free device memory */
cublasFree(device5_X);
cublasFree(device5_Y);

          dgemm_end = gethrtime();
          dgemm4 += (dgemm_end - dgemm_strt);
        }
      }
    }
  }
  prog_end = gethrtime();
  prog = (prog_end - prog_strt);
  ns = ((long )(pow(10,9)));
  nss = ((long )(pow(10,9) * 60));
  printf ("Total program run time :  %.2lf seconds (OR) %.21f min.\n", prog/ns, prog/nss );
  total_dgemm_time = (((dgemm1 + dgemm2) + dgemm3) + dgemm4);
  total_transpose_time = (trans1 + trans2);
  free(B1);
  free(A1_T);
  free(t1);
  free(t2);
  free(C1);
  free(C2_T);
  free(C3_T);
  free(C4_T);
  return 0;
}
