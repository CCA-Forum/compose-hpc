#include "cublas.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
  union __unnamed_class___F0_L22_C5_L49R__L50R__scope____SgSS2___variable_declaration__variable_type__variable_name_L49R__L50R__scope____SgSS2____scope_____DELIMITER__L49R__L50R__scope____SgSS2___variable_declaration__variable_type_hrtime_tUL__typedef_declaration_variable_name_L49R__L50R__scope____SgSS2____scope__ll {
  struct __unnamed_class___F0_L25_C9_L51R_variable_declaration__variable_type_Ui_variable_name_L51R__scope__a__DELIMITER__L51R_variable_declaration__variable_type_Ui_variable_name_L51R__scope__d {
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
  FILE *trans3time;
  FILE *transBtime;
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
  int i;
  int j;
  int k;
  int l;
  long ns;
  long nss;
  int pt;
  int qt;
  int rt;
  int st;
  int p;
  int q;
  int r;
  int s;
  double dif;
  double factor;
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
  double rd;
  double *B1;
  double *B1_T;
  double *C1 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C2 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C3 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C4 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *A1 = (double *)(malloc(((((O * O) * O) * O) * (sizeof(double )))));
  double *T1;
  double *T1_T;
  double *T2;
  double *T2_T;
  double *T3;
  double *T3_T;
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
  T1 = ((double *)(malloc(((((V * O) * O) * O) * (sizeof(double ))))));
  /*% BLAS_TO_CUBLAS prefix=device5 */
  
double *device5_X;
int sizeType_device5 = sizeof(double);

/* Allocate device memory */
cublasAlloc((((V * O) * O) * O), sizeType_device5, (void **)&device5_X);

/* Copy matrix, vectors to device */
cublasSetVector((((V * O) * O) * O), sizeType_device5, T1, 1, device5_X, 1);

/* CUBLAS call */
cublasDscal((((V * O) * O) * O), 0.0, device5_X, 1);

/* Copy result vector back to host */
cublasSetVector((((V * O) * O) * O), sizeType_device5, device5_X, 1, T1, 1);

/* Free device memory */
cublasFree(device5_X);

  dgemm_strt = gethrtime();
  /*% BLAS_TO_CUBLAS prefix=device1 */
  
double *device1_A;
double *device1_B;
double *device1_C;
int sizeType_device1 = sizeof(double);

/* Allocate device memory */
cublasAlloc(V * O, sizeType_device1, (void **)&device1_A);
cublasAlloc(O * ((O * O) * O), sizeType_device1, (void **)&device1_B);
cublasAlloc(V * ((O * O) * O), sizeType_device1, (void **)&device1_C);

/* Copy matrices to device */
cublasSetMatrix(V, O, sizeType_device1, (void *)(C4 + 0), V,
                (void *)device1_A, V);
cublasSetMatrix(O, ((O * O) * O), sizeType_device1, (void *)(A1 + 0), O,
                (void *)device1_B, O);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('T', 'N', V, ((O * O) * O), O, 1.0, device1_A, V, device1_B,
            ((O * O) * O), 1.0, device1_C, ((O * O) * O));

/* Copy result array back to host */
cublasSetMatrix(V, ((O * O) * O), sizeType_device1, (void *)device1_C, V,
                (void *)(T1 + 0), V);

/* Free device memory */
cublasFree(device1_A);
cublasFree(device1_B);
cublasFree(device1_C);

  dgemm_end = gethrtime();
  dgemm1 = (dgemm_end - dgemm_strt);
//printf ("Time taken for dgemm 1 : %.2lf seconds (OR) %.21f min.\n", dgemm1, dgemm1/60 );
  free(A1);
  free(C4);
  T1_T = ((double *)(malloc(((((V * O) * O) * O) * (sizeof(double ))))));
  trans_strt = gethrtime();
  factor = 1.0;
  i = 2;
  j = 1;
  k = 3;
  l = 4;
  tce_sort_4_((T1 + 0),(T1_T + 0),&V,&O,&O,&O,&i,&j,&k,&l,&factor);
  free(T1);
  trans_end = gethrtime();
  trans1 = (trans_end - trans_strt);
//printf ("Time taken for transposing T1 : %.2lf seconds (OR) %.21f min.\n", trans1, trans1/60 );
  T2 = ((double *)(malloc(((((V * V) * O) * O) * (sizeof(double ))))));
  /*% BLAS_TO_CUBLAS prefix=device6 */
  
double *device6_X;
int sizeType_device6 = sizeof(double);

/* Allocate device memory */
cublasAlloc((((V * V) * O) * O), sizeType_device6, (void **)&device6_X);

/* Copy matrix, vectors to device */
cublasSetVector((((V * V) * O) * O), sizeType_device6, T2, 1, device6_X, 1);

/* CUBLAS call */
cublasDscal((((V * V) * O) * O), 0.0, device6_X, 1);

/* Copy result vector back to host */
cublasSetVector((((V * V) * O) * O), sizeType_device6, device6_X, 1, T2, 1);

/* Free device memory */
cublasFree(device6_X);

  dgemm_strt = gethrtime();
  /*% BLAS_TO_CUBLAS prefix=device2 */
  
double *device2_A;
double *device2_B;
double *device2_C;
int sizeType_device2 = sizeof(double);

/* Allocate device memory */
cublasAlloc(V * O, sizeType_device2, (void **)&device2_A);
cublasAlloc(O * ((V * O) * O), sizeType_device2, (void **)&device2_B);
cublasAlloc(V * ((V * O) * O), sizeType_device2, (void **)&device2_C);

/* Copy matrices to device */
cublasSetMatrix(V, O, sizeType_device2, (void *)(C3 + 0), V,
                (void *)device2_A, V);
cublasSetMatrix(O, ((V * O) * O), sizeType_device2, (void *)(T1_T + 0), O,
                (void *)device2_B, O);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('T', 'N', V, ((V * O) * O), O, 1.0, device2_A, V, device2_B,
            ((V * O) * O), 1.0, device2_C, ((V * O) * O));

/* Copy result array back to host */
cublasSetMatrix(V, ((V * O) * O), sizeType_device2, (void *)device2_C, V,
                (void *)(T2 + 0), V);

/* Free device memory */
cublasFree(device2_A);
cublasFree(device2_B);
cublasFree(device2_C);

  dgemm_end = gethrtime();
  dgemm2 = (dgemm_end - dgemm_strt);
//printf ("Time taken for dgemm 2 : %.2lf seconds (OR) %.21f min.\n", dgemm2, dgemm2/60 );
  free(C3);
  free(T1_T);
  T2_T = ((double *)(malloc(((((O * V) * V) * O) * (sizeof(double ))))));
  trans_strt = gethrtime();
  factor = 1.0;
  i = 3;
  j = 1;
  k = 2;
  l = 4;
  tce_sort_4_((T2 + 0),(T2_T + 0),&V,&V,&O,&O,&i,&j,&k,&l,&factor);
  free(T2);
  trans_end = gethrtime();
  trans2 = (trans_end - trans_strt);
  T3 = ((double *)(malloc(((((V * V) * V) * O) * (sizeof(double ))))));
  /*% BLAS_TO_CUBLAS prefix=device7 */
  
double *device7_X;
int sizeType_device7 = sizeof(double);

/* Allocate device memory */
cublasAlloc((((V * V) * V) * O), sizeType_device7, (void **)&device7_X);

/* Copy matrix, vectors to device */
cublasSetVector((((V * V) * V) * O), sizeType_device7, T3, 1, device7_X, 1);

/* CUBLAS call */
cublasDscal((((V * V) * V) * O), 0.0, device7_X, 1);

/* Copy result vector back to host */
cublasSetVector((((V * V) * V) * O), sizeType_device7, device7_X, 1, T3, 1);

/* Free device memory */
cublasFree(device7_X);

  dgemm_strt = gethrtime();
  /*% BLAS_TO_CUBLAS prefix=device3 */
  
double *device3_A;
double *device3_B;
double *device3_C;
int sizeType_device3 = sizeof(double);

/* Allocate device memory */
cublasAlloc(V * O, sizeType_device3, (void **)&device3_A);
cublasAlloc(O * ((V * V) * O), sizeType_device3, (void **)&device3_B);
cublasAlloc(V * ((V * V) * O), sizeType_device3, (void **)&device3_C);

/* Copy matrices to device */
cublasSetMatrix(V, O, sizeType_device3, (void *)(C2 + 0), V,
                (void *)device3_A, V);
cublasSetMatrix(O, ((V * V) * O), sizeType_device3, (void *)(T2_T + 0), O,
                (void *)device3_B, O);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('T', 'N', V, ((V * V) * O), O, 1.0, device3_A, V, device3_B,
            ((V * V) * O), 1.0, device3_C, ((V * V) * O));

/* Copy result array back to host */
cublasSetMatrix(V, ((V * V) * O), sizeType_device3, (void *)device3_C, V,
                (void *)(T3 + 0), V);

/* Free device memory */
cublasFree(device3_A);
cublasFree(device3_B);
cublasFree(device3_C);

  dgemm_end = gethrtime();
  dgemm3 = (dgemm_end - dgemm_strt);
  free(T2_T);
  free(C2);
  T3_T = ((double *)(malloc(((((O * V) * V) * V) * (sizeof(double ))))));
  trans_strt = gethrtime();
  factor = 1.0;
  i = 4;
  j = 2;
  k = 1;
  l = 3;
  tce_sort_4_((T3 + 0),(T3_T + 0),&V,&V,&V,&O,&i,&j,&k,&l,&factor);
  free(T3);
  trans_end = gethrtime();
  trans3 = (trans_end - trans_strt);
//printf ("Time taken for transposing T3 : %.2lf seconds (OR) %.21f min.\n", trans3, trans3/60 );
  B1 = ((double *)(malloc(((((V * V) * V) * V) * (sizeof(double ))))));
  /*% BLAS_TO_CUBLAS prefix=device8 */
  
double *device8_X;
int sizeType_device8 = sizeof(double);

/* Allocate device memory */
cublasAlloc((((V * V) * V) * V), sizeType_device8, (void **)&device8_X);

/* Copy matrix, vectors to device */
cublasSetVector((((V * V) * V) * V), sizeType_device8, B1, 1, device8_X, 1);

/* CUBLAS call */
cublasDscal((((V * V) * V) * V), 0.0, device8_X, 1);

/* Copy result vector back to host */
cublasSetVector((((V * V) * V) * V), sizeType_device8, device8_X, 1, B1, 1);

/* Free device memory */
cublasFree(device8_X);

  dgemm_strt = gethrtime();
  /*% BLAS_TO_CUBLAS prefix=device4 */
  
double *device4_A;
double *device4_B;
double *device4_C;
int sizeType_device4 = sizeof(double);

/* Allocate device memory */
cublasAlloc(V * O, sizeType_device4, (void **)&device4_A);
cublasAlloc(O * ((V * V) * V), sizeType_device4, (void **)&device4_B);
cublasAlloc(V * ((V * V) * V), sizeType_device4, (void **)&device4_C);

/* Copy matrices to device */
cublasSetMatrix(V, O, sizeType_device4, (void *)(C1 + 0), V,
                (void *)device4_A, V);
cublasSetMatrix(O, ((V * V) * V), sizeType_device4, (void *)(T3_T + 0), O,
                (void *)device4_B, O);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('T', 'N', V, ((V * V) * V), O, 1.0, device4_A, V, device4_B,
            ((V * V) * V), 1.0, device4_C, ((V * V) * V));

/* Copy result array back to host */
cublasSetMatrix(V, ((V * V) * V), sizeType_device4, (void *)device4_C, V,
                (void *)(B1 + 0), V);

/* Free device memory */
cublasFree(device4_A);
cublasFree(device4_B);
cublasFree(device4_C);

  dgemm_end = gethrtime();
  dgemm4 = (dgemm_end - dgemm_strt);
//printf ("Time taken for dgemm 4 : %.2lf seconds (OR) %.21f min.\n", dgemm4, dgemm4/60 );
  free(T3_T);
  free(C1);
  B1_T = ((double *)(malloc(((((V * V) * V) * V) * (sizeof(double ))))));
  trans_strt = gethrtime();
  factor = 1.0;
  i = 4;
  j = 2;
  k = 3;
  l = 1;
  tce_sort_4_((B1 + 0),(B1_T + 0),&V,&V,&V,&V,&i,&j,&k,&l,&factor);
  free(B1);
  trans_end = gethrtime();
  transB = (trans_end - trans_strt);
  prog_end = gethrtime();
  prog = (prog_end - prog_strt);
  ns = ((long )(pow(10,9)));
  nss = ((long )(pow(10,9) * 60));
//printf ("Total program run time :  %.2lf seconds (OR) %.21f min.\n", prog/ns, prog/nss );
  total_dgemm_time = (((dgemm1 + dgemm2) + dgemm3) + dgemm4);
  total_transpose_time = (((trans1 + trans2) + trans3) + transB);
  free(B1_T);
  return 1;
}
