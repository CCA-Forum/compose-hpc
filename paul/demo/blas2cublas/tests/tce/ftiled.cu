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
#endif
typedef unsigned long long hrtime_t;

inline hrtime_t gethrtime()
{
  union __unnamed_class___F0_L22_C5_L50R__L51R__scope____SgSS2___variable_declaration__variable_type__variable_name_L50R__L51R__scope____SgSS2____scope_____DELIMITER__L50R__L51R__scope____SgSS2___variable_declaration__variable_type_hrtime_tUL__typedef_declaration_variable_name_L50R__L51R__scope____SgSS2____scope__ll {
  struct __unnamed_class___F0_L25_C9_L52R_variable_declaration__variable_type_Ui_variable_name_L52R__scope__a__DELIMITER__L52R_variable_declaration__variable_type_Ui_variable_name_L52R__scope__d {
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
  FILE *transCtime;
  FILE *inittime;
  FILE *copy1time;
  FILE *copy2time;
  FILE *totaldgemmtime;
  FILE *totaltranstime;
  FILE *totalcopytime;
  int pt;
  int qt;
  int rt;
  int st;
  int a;
  int b;
  int c;
  int d;
  int V = atoi(argv[1]);
  int O = atoi(argv[2]);
  int T = atoi(argv[3]);
  int p;
  int q;
  int r;
  int s;
  int ss;
  int pp;
  int i;
  int j;
  int k;
  int l;
  int OT = (O * T);
  int O2 = (O * O);
  int O3 = (O2 * O);
  int V2 = (V * V);
  int V3 = (V2 * V);
  int V2T = ((V * V) * T);
  int T2 = (T * T);
  int VT2 = (V * T2);
  int T3 = (T2 * T);
  int O2T = (OT * O);
  int OT2 = (OT * T);
  int T4 = (T2 * T2);
  long ns;
  long nss;
  int lt1;
  int lt2;
  int lt3;
  int lt4;
  int tile1;
  int tile2;
  int dim1;
  int dim2;
  int dim3;
  int dim4;
  int alloc;
  int index;
  int b1;
  int b2;
  int b3;
  int b4;
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
  double transC;
  double prog;
  double *B1;
  double *B1_T;
  double *A1_C;
  double *A1 = (double *)(malloc(((((O * O) * O) * O) * (sizeof(double )))));
  double *C1 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C2 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C3 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C4 = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C1_T = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C2_T = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C3_T = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C4_T = (double *)(malloc(((O * V) * (sizeof(double )))));
  double *C1_C;
  double *t1;
  double *t2;
  double *t3;
  double *t1_t;
  double *t2_t;
  double *t3_t;
  ((prog_strt = 0) , (array_init_strt = 0));
  ((((((dgemm1 = 0) , (dgemm2 = 0))) , (dgemm3 = 0))) , (dgemm4 = 0));
  ((((((trans1 = 0) , (trans2 = 0))) , (trans3 = 0))) , (transB = 0));
  ((((((transC = 0) , (copy1 = 0))) , (copy2 = 0))) , (copy3 = 0));
//printf("V=%d, O=%d, T=%d\n",V,O,T);
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
  tce_sort_2_((C1 + 0),(C1_T + 0),&O,&V,&i,&j,&factor);
  free(C1);
  free(C2);
  free(C3);
  free(C4);
  trans_end = gethrtime();
  transC = (trans_end - trans_strt);
  B1 = ((double *)(malloc(((V3 * V) * (sizeof(double ))))));
  /*% BLAS_TO_CUBLAS prefix=device5 */
  
double *device5_X;
int sizeType_device5 = sizeof(double);

/* Allocate device memory */
cublasAlloc((V3 * V), sizeType_device5, (void **)&device5_X);

/* Copy matrix, vectors to device */
cublasSetVector((V3 * V), sizeType_device5, B1, 1, device5_X, 1);

/* CUBLAS call */
cublasDscal((V3 * V), 0.0, device5_X, 1);

/* Copy result vector back to host */
cublasSetVector((V3 * V), sizeType_device5, device5_X, 1, B1, 1);

/* Free device memory */
cublasFree(device5_X);

  for (s = 0; s < O; s = (s + T)) {
    b4 = 0;
    if ((s + T) > O) {
      lt4 = (O - s);
      b4 = 1;
    }
    if (b4 == 1) {
      tile1 = lt4;
      A1_C = ((double *)(malloc(((((O * O) * O) * lt4) * (sizeof(double ))))));
    }
    else {
      tile1 = T;
      A1_C = ((double *)(malloc(((((O * O) * O) * T) * (sizeof(double ))))));
    }
    copy_strt = gethrtime();
    for (pt = 0; pt < O; pt = (pt + T)) {
      for (qt = 0; qt < O; qt = (qt + T)) {
        for (rt = 0; rt < O; rt = (rt + T)) {
          for (p = pt; p < ((((pt + T) < O)?(pt + T) : O)); p++) {
            for (q = qt; q < ((((qt + T) < O)?(qt + T) : O)); q++) {
              for (r = rt; r < ((((rt + T) < O)?(rt + T) : O)); r++) {
                /*% BLAS_TO_CUBLAS prefix=device9 */
                
double *device9_X;
double *device9_Y;
int sizeType_device9 = sizeof(double);

/* Allocate device memory */
cublasAlloc(tile1, sizeType_device9, (void **)&device9_X);
cublasAlloc(tile1, sizeType_device9, (void **)&device9_Y);

/* Copy matrix, vectors to device */
cublasSetVector(tile1, sizeType_device9,
                (A1 + ((((((p * O) + q) * O) + r) * O) + s)), 1, device9_X, 1);

/* CUBLAS call */
cublasDcopy(tile1, device9_X, 1, device9_Y, 1);

/* Copy result vector back to host */
cublasSetVector(tile1, sizeType_device9, device9_Y, 1,
                (A1_C + (((((p * O) + q) * O) + r) * tile1)), 1);

/* Free device memory */
cublasFree(device9_X);
cublasFree(device9_Y);

              }
            }
          }
        }
      }
    }
    copy_end = gethrtime();
    copy1 += (copy_end - copy_strt);
    copy_strt = gethrtime();
    if (b4 == 1) {
      tile1 = lt4;
      C1_C = ((double *)(malloc(((tile1 * V) * (sizeof(double ))))));
    }
    else {
      tile1 = T;
      C1_C = ((double *)(malloc(((tile1 * V) * (sizeof(double ))))));
    }
    for (pp = 0; pp < V; pp++) {
      /*% BLAS_TO_CUBLAS prefix=device10 */
      
double *device10_X;
double *device10_Y;
int sizeType_device10 = sizeof(double);

/* Allocate device memory */
cublasAlloc(tile1, sizeType_device10, (void **)&device10_X);
cublasAlloc(tile1, sizeType_device10, (void **)&device10_Y);

/* Copy matrix, vectors to device */
cublasSetVector(tile1, sizeType_device10, (C1_T + ((pp * O) + s)), 1,
                device10_X, 1);

/* CUBLAS call */
cublasDcopy(tile1, device10_X, 1, device10_Y, 1);

/* Copy result vector back to host */
cublasSetVector(tile1, sizeType_device10, device10_Y, 1,
                (C1_C + (pp * tile1)), 1);

/* Free device memory */
cublasFree(device10_X);
cublasFree(device10_Y);

    }
    copy_end = gethrtime();
    copy2 += (copy_end - copy_strt);
    for (a = 0; a < V; a = (a + T)) {
      b1 = 0;
      if ((a + T) > V) {
        lt1 = (V - a);
        b1 = 1;
      }
      if ((b4 == 1) && (b1 == 1)) {
        alloc = ((O2 * lt1) * lt4);
      }
      else if (b1 == 1) {
        alloc = (O2T * lt1);
      }
      else if (b4 == 1) {
        alloc = (O2T * lt4);
      }
      else {
        alloc = (O2 * T2);
      }
      t1 = ((double *)(malloc((alloc * (sizeof(double ))))));
      t1_t = ((double *)(malloc((alloc * (sizeof(double ))))));
      /*% BLAS_TO_CUBLAS prefix=device6 */
      
double *device6_X;
int sizeType_device6 = sizeof(double);

/* Allocate device memory */
cublasAlloc(alloc, sizeType_device6, (void **)&device6_X);

/* Copy matrix, vectors to device */
cublasSetVector(alloc, sizeType_device6, t1, 1, device6_X, 1);

/* CUBLAS call */
cublasDscal(alloc, 0.0, device6_X, 1);

/* Copy result vector back to host */
cublasSetVector(alloc, sizeType_device6, device6_X, 1, t1, 1);

/* Free device memory */
cublasFree(device6_X);

      dgemm_strt = gethrtime();
      if ((b4 == 1) && (b1 == 1)) {
        tile1 = lt1;
        tile2 = (O2 * lt4);
      }
      else if (b1 == 1) {
        tile1 = lt1;
        tile2 = O2T;
      }
      else if (b4 == 1) {
        tile1 = T;
        tile2 = (O2 * lt4);
      }
      else {
        tile1 = T;
        tile2 = O2T;
      }
      /*% BLAS_TO_CUBLAS prefix=device1 */
      
double *device1_A;
double *device1_B;
double *device1_C;
int sizeType_device1 = sizeof(double);

/* Allocate device memory */
cublasAlloc(tile1 * O, sizeType_device1, (void **)&device1_A);
cublasAlloc(O * tile2, sizeType_device1, (void **)&device1_B);
cublasAlloc(tile1 * tile2, sizeType_device1, (void **)&device1_C);

/* Copy matrices to device */
cublasSetMatrix(tile1, O, sizeType_device1, (void *)(C4_T + (a * O)), tile1,
                (void *)device1_A, tile1);
cublasSetMatrix(O, tile2, sizeType_device1, (void *)(A1_C + 0), O,
                (void *)device1_B, O);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('N', 'N', tile1, tile2, O, 1.0, device1_A, O, device1_B, tile2,
            1.0, device1_C, tile2);

/* Copy result array back to host */
cublasSetMatrix(tile1, tile2, sizeType_device1, (void *)device1_C, tile1,
                (void *)(t1 + 0), tile1);

/* Free device memory */
cublasFree(device1_A);
cublasFree(device1_B);
cublasFree(device1_C);

      dgemm_end = gethrtime();
      dgemm1 += (dgemm_end - dgemm_strt);
      trans_strt = gethrtime();
      factor = 1.0;
      i = 2;
      j = 3;
      k = 4;
      l = 1;
      if ((b4 == 1) && (b1 == 1)) {
        dim1 = lt1;
        dim2 = O;
        dim3 = O;
        dim4 = lt4;
      }
      else if (b4 == 1) {
        dim1 = T;
        dim2 = O;
        dim3 = O;
        dim4 = lt4;
      }
      else if (b1 == 1) {
        dim1 = lt1;
        dim2 = O;
        dim3 = O;
        dim4 = T;
      }
      else {
        dim1 = T;
        dim2 = O;
        dim3 = O;
        dim4 = T;
      }
      tce_sort_4_((t1 + 0),(t1_t + 0),&dim1,&dim2,&dim3,&dim4,&i,&j,&k,&l,&factor);
      free(t1);
      trans_end = gethrtime();
      trans1 += (trans_end - trans_strt);
      for (b = 0; b < V; b = (b + T)) {
        b2 = 0;
        if ((b + T) > V) {
          lt2 = (V - b);
          b2 = 1;
        }
        if (((b4 == 1) && (b1 == 1)) && (b2 == 1)) {
          alloc = (((O * lt1) * lt2) * lt4);
        }
        else if ((b4 == 1) && (b1 == 1)) {
          alloc = ((OT * lt1) * lt4);
        }
        else if ((b4 == 1) && (b2 == 1)) {
          alloc = ((OT * lt2) * lt4);
        }
        else if ((b1 == 1) && (b2 == 1)) {
          alloc = ((OT * lt2) * lt1);
        }
        else if (b1 == 1) {
          alloc = (OT2 * lt1);
        }
        else if (b2 == 1) {
          alloc = (OT2 * lt2);
        }
        else if (b4 == 1) {
          alloc = (OT2 * lt4);
        }
        else {
          alloc = (O * T3);
        }
        t2 = ((double *)(malloc((alloc * (sizeof(double ))))));
        t2_t = ((double *)(malloc((alloc * (sizeof(double ))))));
        /*% BLAS_TO_CUBLAS prefix=device7 */
        
double *device7_X;
int sizeType_device7 = sizeof(double);

/* Allocate device memory */
cublasAlloc(alloc, sizeType_device7, (void **)&device7_X);

/* Copy matrix, vectors to device */
cublasSetVector(alloc, sizeType_device7, t2, 1, device7_X, 1);

/* CUBLAS call */
cublasDscal(alloc, 0.0, device7_X, 1);

/* Copy result vector back to host */
cublasSetVector(alloc, sizeType_device7, device7_X, 1, t2, 1);

/* Free device memory */
cublasFree(device7_X);

        dgemm_strt = gethrtime();
        if (((b4 == 1) && (b1 == 1)) && (b2 == 1)) {
          tile1 = lt2;
          tile2 = ((O * lt1) * lt4);
        }
        else if ((b4 == 1) && (b1 == 1)) {
          tile1 = T;
          tile2 = ((O * lt1) * lt4);
        }
        else if ((b4 == 1) && (b2 == 1)) {
          tile1 = lt2;
          tile2 = (OT * lt4);
        }
        else if ((b1 == 1) && (b2 == 1)) {
          tile1 = lt2;
          tile2 = (OT * lt1);
        }
        else if (b1 == 1) {
          tile1 = T;
          tile2 = (OT * lt1);
        }
        else if (b2 == 1) {
          tile1 = lt2;
          tile2 = OT2;
        }
        else if (b4 == 1) {
          tile1 = T;
          tile2 = (OT * lt4);
        }
        else {
          tile1 = T;
          tile2 = OT2;
        }
        /*% BLAS_TO_CUBLAS prefix=device2 */
        
double *device2_A;
double *device2_B;
double *device2_C;
int sizeType_device2 = sizeof(double);

/* Allocate device memory */
cublasAlloc(tile1 * O, sizeType_device2, (void **)&device2_A);
cublasAlloc(O * tile2, sizeType_device2, (void **)&device2_B);
cublasAlloc(tile1 * tile2, sizeType_device2, (void **)&device2_C);

/* Copy matrices to device */
cublasSetMatrix(tile1, O, sizeType_device2, (void *)(C3_T + (b * O)), tile1,
                (void *)device2_A, tile1);
cublasSetMatrix(O, tile2, sizeType_device2, (void *)(t1_t + 0), O,
                (void *)device2_B, O);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('N', 'N', tile1, tile2, O, 1.0, device2_A, O, device2_B, tile2,
            1.0, device2_C, tile2);

/* Copy result array back to host */
cublasSetMatrix(tile1, tile2, sizeType_device2, (void *)device2_C, tile1,
                (void *)(t2 + 0), tile1);

/* Free device memory */
cublasFree(device2_A);
cublasFree(device2_B);
cublasFree(device2_C);

        dgemm_end = gethrtime();
        dgemm2 += (dgemm_end - dgemm_strt);
        trans_strt = gethrtime();
        factor = 1.0;
        i = 2;
        j = 3;
        k = 4;
        l = 1;
        if (((b4 == 1) && (b1 == 1)) && (b2 == 1)) {
          dim1 = lt2;
          dim2 = O;
          dim3 = lt4;
          dim4 = lt1;
        }
        else if ((b4 == 1) && (b1 == 1)) {
          dim1 = T;
          dim2 = O;
          dim3 = lt4;
          dim4 = lt1;
        }
        else if ((b4 == 1) && (b2 == 1)) {
          dim1 = lt2;
          dim2 = O;
          dim3 = lt4;
          dim4 = T;
        }
        else if ((b1 == 1) && (b2 == 1)) {
          dim1 = lt2;
          dim2 = O;
          dim3 = T;
          dim4 = lt1;
        }
        else if (b1 == 1) {
          dim1 = T;
          dim2 = O;
          dim3 = T;
          dim4 = lt1;
        }
        else if (b2 == 1) {
          dim1 = lt2;
          dim2 = O;
          dim3 = T;
          dim4 = T;
        }
        else if (b4 == 1) {
          dim1 = T;
          dim2 = O;
          dim3 = lt4;
          dim4 = T;
        }
        else {
          dim1 = T;
          dim2 = O;
          dim3 = T;
          dim4 = T;
        }
        tce_sort_4_((t2 + 0),(t2_t + 0),&dim1,&dim2,&dim3,&dim4,&i,&j,&k,&l,&factor);
        free(t2);
        trans_end = gethrtime();
        trans2 += (trans_end - trans_strt);
        for (c = 0; c < V; c = (c + T)) {
          b3 = 0;
          if ((c + T) > V) {
            lt3 = (V - c);
            b3 = 1;
          }
          if ((((b4 == 1) && (b1 == 1)) && (b2 == 1)) && (b3 == 1)) 
            alloc = (((lt1 * lt2) * lt3) * lt4);
          else if (((b1 == 1) && (b2 == 1)) && (b3 == 1)) 
            alloc = (((T * lt1) * lt2) * lt3);
          else if (((b1 == 1) && (b2 == 1)) && (b4 == 1)) 
            alloc = (((T * lt1) * lt2) * lt4);
          else if (((b1 == 1) && (b3 == 1)) && (b4 == 1)) 
            alloc = (((T * lt4) * lt1) * lt3);
          else if (((b2 == 1) && (b3 == 1)) && (b4 == 1)) 
            alloc = (((T * lt4) * lt2) * lt3);
          else if ((b1 == 1) && (b2 == 1)) 
            alloc = ((T2 * lt1) * lt2);
          else if ((b1 == 1) && (b3 == 1)) 
            alloc = ((T2 * lt1) * lt3);
          else if ((b1 == 1) && (b4 == 1)) 
            alloc = ((T2 * lt1) * lt4);
          else if ((b2 == 1) && (b3 == 1)) 
            alloc = ((T2 * lt2) * lt3);
          else if ((b2 == 1) && (b4 == 1)) 
            alloc = ((T2 * lt2) * lt4);
          else if ((b3 == 1) && (b4 == 1)) 
            alloc = ((T2 * lt3) * lt4);
          else if (b1 == 1) 
            alloc = (T3 * lt1);
          else if (b2 == 1) 
            alloc = (T3 * lt2);
          else if (b3 == 1) 
            alloc = (T3 * lt3);
          else if (b4 == 1) 
            alloc = (T3 * lt4);
          else 
            alloc = T4;
          t3 = ((double *)(malloc((alloc * (sizeof(double ))))));
          t3_t = ((double *)(malloc((alloc * (sizeof(double ))))));
          /*% BLAS_TO_CUBLAS prefix=device8 */
          
double *device8_X;
int sizeType_device8 = sizeof(double);

/* Allocate device memory */
cublasAlloc(alloc, sizeType_device8, (void **)&device8_X);

/* Copy matrix, vectors to device */
cublasSetVector(alloc, sizeType_device8, t3, 1, device8_X, 1);

/* CUBLAS call */
cublasDscal(alloc, 0.0, device8_X, 1);

/* Copy result vector back to host */
cublasSetVector(alloc, sizeType_device8, device8_X, 1, t3, 1);

/* Free device memory */
cublasFree(device8_X);

          dgemm_strt = gethrtime();
          if ((((b4 == 1) && (b1 == 1)) && (b2 == 1)) && (b3 == 1)) {
            tile1 = lt3;
            tile2 = ((lt1 * lt2) * lt4);
          }
          else if (((b1 == 1) && (b2 == 1)) && (b3 == 1)) {
            tile1 = lt3;
            tile2 = ((T * lt1) * lt2);
          }
          else if (((b1 == 1) && (b2 == 1)) && (b4 == 1)) {
            tile1 = T;
            tile2 = ((lt1 * lt2) * lt4);
          }
          else if (((b1 == 1) && (b3 == 1)) && (b4 == 1)) {
            tile1 = lt3;
            tile2 = ((T * lt1) * lt4);
          }
          else if (((b2 == 1) && (b3 == 1)) && (b4 == 1)) {
            tile1 = lt3;
            tile2 = ((T * lt2) * lt4);
          }
          else if ((b1 == 1) && (b2 == 1)) {
            tile1 = T;
            tile2 = ((T * lt1) * lt2);
          }
          else if ((b1 == 1) && (b3 == 1)) {
            tile1 = lt3;
            tile2 = (T2 * lt1);
          }
          else if ((b1 == 1) && (b4 == 1)) {
            tile1 = T;
            tile2 = ((T * lt1) * lt4);
          }
          else if ((b2 == 1) && (b3 == 1)) {
            tile1 = lt3;
            tile2 = (T2 * lt2);
          }
          else if ((b2 == 1) && (b4 == 1)) {
            tile1 = T;
            tile2 = ((T * lt2) * lt4);
          }
          else if ((b3 == 1) && (b4 == 1)) {
            tile1 = lt3;
            tile2 = (T2 * lt4);
          }
          else if (b1 == 1) {
            tile1 = T;
            tile2 = (T2 * lt1);
          }
          else if (b2 == 1) {
            tile1 = T;
            tile2 = (T2 * lt2);
          }
          else if (b3 == 1) {
            tile1 = lt3;
            tile2 = T3;
          }
          else if (b4 == 1) {
            tile1 = T;
            tile2 = (T2 * lt4);
          }
          else {
            tile1 = T;
            tile2 = T3;
          }
          /*% BLAS_TO_CUBLAS prefix=device3 */
          
double *device3_A;
double *device3_B;
double *device3_C;
int sizeType_device3 = sizeof(double);

/* Allocate device memory */
cublasAlloc(tile1 * O, sizeType_device3, (void **)&device3_A);
cublasAlloc(O * tile2, sizeType_device3, (void **)&device3_B);
cublasAlloc(tile1 * tile2, sizeType_device3, (void **)&device3_C);

/* Copy matrices to device */
cublasSetMatrix(tile1, O, sizeType_device3, (void *)(C2_T + (c * O)), tile1,
                (void *)device3_A, tile1);
cublasSetMatrix(O, tile2, sizeType_device3, (void *)(t2_t + 0), O,
                (void *)device3_B, O);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('N', 'N', tile1, tile2, O, 1.0, device3_A, O, device3_B, tile2,
            1.0, device3_C, tile2);

/* Copy result array back to host */
cublasSetMatrix(tile1, tile2, sizeType_device3, (void *)device3_C, tile1,
                (void *)(t3 + 0), tile1);

/* Free device memory */
cublasFree(device3_A);
cublasFree(device3_B);
cublasFree(device3_C);

          dgemm_end = gethrtime();
          dgemm3 += (dgemm_end - dgemm_strt);
          trans_strt = gethrtime();
          factor = 1.0;
{
            i = 2;
            j = 3;
            k = 4;
            l = 1;
          }
          if ((((b4 == 1) && (b1 == 1)) && (b2 == 1)) && (b3 == 1)) {
            dim1 = lt3;
            dim2 = lt4;
            dim3 = lt1;
            dim4 = lt2;
          }
          else if (((b1 == 1) && (b2 == 1)) && (b3 == 1)) {
            dim1 = lt3;
            dim2 = T;
            dim3 = lt1;
            dim4 = lt2;
          }
          else if (((b1 == 1) && (b2 == 1)) && (b4 == 1)) {
            dim1 = T;
            dim2 = lt4;
            dim3 = lt1;
            dim4 = lt2;
          }
          else if (((b1 == 1) && (b3 == 1)) && (b4 == 1)) {
            dim1 = lt3;
            dim2 = lt4;
            dim3 = lt1;
            dim4 = T;
          }
          else if (((b2 == 1) && (b3 == 1)) && (b4 == 1)) {
            dim1 = lt3;
            dim2 = lt4;
            dim3 = T;
            dim4 = lt2;
          }
          else if ((b1 == 1) && (b2 == 1)) {
            dim1 = T;
            dim2 = T;
            dim3 = lt1;
            dim4 = lt2;
          }
          else if ((b1 == 1) && (b3 == 1)) {
            dim1 = lt3;
            dim2 = T;
            dim3 = lt1;
            dim4 = T;
          }
          else if ((b1 == 1) && (b4 == 1)) {
            dim1 = T;
            dim2 = lt4;
            dim3 = lt1;
            dim4 = T;
          }
          else if ((b2 == 1) && (b3 == 1)) {
            dim1 = lt3;
            dim2 = T;
            dim3 = T;
            dim4 = lt2;
          }
          else if ((b2 == 1) && (b4 == 1)) {
            dim1 = T;
            dim2 = lt4;
            dim3 = T;
            dim4 = lt2;
          }
          else if ((b3 == 1) && (b4 == 1)) {
            dim1 = lt3;
            dim2 = lt4;
            dim3 = T;
            dim4 = T;
          }
          else if (b1 == 1) {
            dim1 = T;
            dim2 = T;
            dim3 = lt1;
            dim4 = T;
          }
          else if (b2 == 1) {
            dim1 = T;
            dim2 = T;
            dim3 = T;
            dim4 = lt2;
          }
          else if (b3 == 1) {
            dim1 = lt3;
            dim2 = T;
            dim3 = T;
            dim4 = T;
          }
          else if (b4 == 1) {
            dim1 = T;
            dim2 = lt4;
            dim3 = T;
            dim4 = T;
          }
          else {
            dim1 = T;
            dim2 = T;
            dim3 = T;
            dim4 = T;
          }
          tce_sort_4_((t3 + 0),(t3_t + 0),&dim1,&dim2,&dim3,&dim4,&i,&j,&k,&l,&factor);
          free(t3);
          trans_end = gethrtime();
          trans3 += (trans_end - trans_strt);
          dgemm_strt = gethrtime();
          if (((O % T) == 0) && ((V % T) == 0)) {
            tile1 = T;
            tile2 = T3;
            index = (((a * V3) + (b * V2T)) + (c * VT2));
          }
          else {
            if ((((b4 == 1) && (b1 == 1)) && (b2 == 1)) && (b3 == 1)) {
              tile1 = lt4;
              tile2 = ((lt1 * lt2) * lt3);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * lt2) * lt3));
            }
            else if (((b1 == 1) && (b2 == 1)) && (b3 == 1)) {
              tile1 = T;
              tile2 = ((lt1 * lt2) * lt3);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * lt2) * lt3));
            }
            else if (((b1 == 1) && (b2 == 1)) && (b4 == 1)) {
              tile1 = lt4;
              tile2 = ((lt1 * lt2) * T);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * lt1) * lt2));
            }
            else if ((b1 == 1) && (b2 == 1)) {
              tile1 = T;
              tile2 = ((lt1 * lt2) * T);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * lt1) * lt2));
            }
            else if (((b1 == 1) && (b3 == 1)) && (b4 == 1)) {
              tile1 = lt4;
              tile2 = ((lt1 * T) * lt3);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * T) * lt3));
            }
            else if ((b1 == 1) && (b3 == 1)) {
              tile1 = T;
              tile2 = ((lt1 * T) * lt3);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * T) * lt3));
            }
            else if (((b2 == 1) && (b3 == 1)) && (b4 == 1)) {
              tile1 = lt4;
              tile2 = ((T * lt2) * lt3);
              index = (((a * V3) + (b * V2T)) + (((c * V) * T) * lt3));
            }
            else if ((b2 == 1) && (b3 == 1)) {
              tile1 = T;
              tile2 = ((T * lt2) * lt3);
              index = (((a * V3) + (b * V2T)) + (((c * V) * T) * lt3));
            }
            else if ((b1 == 1) && (b4 == 1)) {
              tile1 = lt4;
              tile2 = (lt1 * T2);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * T) * lt1));
            }
            else if (b1 == 1) {
              tile1 = T;
              tile2 = (lt1 * T2);
              index = (((a * V3) + ((b * V2) * lt1)) + (((c * V) * T) * lt1));
            }
            else if ((b2 == 1) && (b4 == 1)) {
              tile1 = lt4;
              tile2 = (lt2 * T2);
              index = (((a * V3) + (b * V2T)) + (((c * V) * T) * lt2));
            }
            else if (b2 == 1) {
              tile1 = T;
              tile2 = (lt2 * T2);
              index = (((a * V3) + (b * V2T)) + (((c * V) * T) * lt2));
            }
            else if ((b3 == 1) && (b4 == 1)) {
              tile1 = lt4;
              tile2 = (T2 * lt3);
              index = (((a * V3) + (b * V2T)) + (c * VT2));
            }
            else if (b3 == 1) {
              tile1 = T;
              tile2 = (lt3 * T2);
              index = (((a * V3) + (b * V2T)) + (c * VT2));
            }
            else if (b4 == 1) {
              tile1 = lt4;
              tile2 = T3;
              index = (((a * V3) + (b * V2T)) + (c * VT2));
            }
            else {
              tile1 = T;
              tile2 = T3;
              index = (((a * V3) + (b * V2T)) + (c * VT2));
            }
          }
          /*% BLAS_TO_CUBLAS prefix=device4 */
          
double *device4_A;
double *device4_B;
double *device4_C;
int sizeType_device4 = sizeof(double);

/* Allocate device memory */
cublasAlloc(V * tile1, sizeType_device4, (void **)&device4_A);
cublasAlloc(tile1 * tile2, sizeType_device4, (void **)&device4_B);
cublasAlloc(V * tile2, sizeType_device4, (void **)&device4_C);

/* Copy matrices to device */
cublasSetMatrix(V, tile1, sizeType_device4, (void *)(C1_C + 0), V,
                (void *)device4_A, V);
cublasSetMatrix(tile1, tile2, sizeType_device4, (void *)(t3_t + 0), tile1,
                (void *)device4_B, tile1);

/* CUBLAS call */
//BLAS_TO_CUBLAS transformation performance warning: 
//CUBLAS calls assume arrays are stored in column-major order. 
//The original BLAS call specified that the arrays are stored in row-major order. 
cublasDgemm('N', 'N', V, tile2, tile1, 1.0, device4_A, tile1, device4_B,
            tile2, 1.0, device4_C, tile2);

/* Copy result array back to host */
cublasSetMatrix(V, tile2, sizeType_device4, (void *)device4_C, V,
                (void *)(B1 + index), V);

/* Free device memory */
cublasFree(device4_A);
cublasFree(device4_B);
cublasFree(device4_C);

          dgemm_end = gethrtime();
          dgemm4 += (dgemm_end - dgemm_strt);
          free(t3_t);
        }
        free(t2_t);
      }
      free(t1_t);
    }
    free(A1_C);
    free(C1_C);
  }
  free(A1);
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
//printf ("Total program run time :    %.21f seconds (OR)   %.21f min.\n", prog/ns, prog/nss );
  total_dgemm_time = (((dgemm1 + dgemm2) + dgemm3) + dgemm4);
  total_transpose_time = ((((trans1 + trans2) + trans3) + transB) + transC);
  total_copy = (copy1 + copy2);
  free(B1_T);
  free(C1_T);
  free(C2_T);
  free(C3_T);
  free(C4_T);
  return 0;
}
