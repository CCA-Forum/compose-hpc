#ifndef BLAS2CUBLAS_H_
#define BLAS2CUBLAS_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include "rose.h"
using namespace std;

void handleGEMM(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleSYHEMM(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleSYHERK(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleSYHER2K(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleTRSMM(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleGBMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleGEMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleGER(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleHSBMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleHSEYMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleHESYR2(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleHESYR(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleHSPMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleHSPR2(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleHSPR(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleTBSMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleTPSMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleTRSMV(ofstream &, bool, bool, string, string, SgExprListExp*, int*);
void handleSumNrm2Aminmax(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleAXPY(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleAXPBY(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleCOPY(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleDOT(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleSCAL(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleSWAP(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleROTM(ofstream &, bool, string, string, SgExprListExp*, int*);
void handleROT(ofstream &, bool, string, string, SgExprListExp*, int*);

inline void RowMajorWarning(ostringstream &cocciFptr, bool warnRowMajor) {
    if (warnRowMajor) {
        cocciFptr << "+  //BLAS_TO_CUBLAS transformation warning: \n";
        cocciFptr
        << "+  //CUBLAS calls assume arrays are stored in column-major order. \n";
        cocciFptr
        << "+  //The original BLAS call specified that the arrays are stored in row-major order. \n";
    }
}

inline void FreeDeviceMemoryB3(ostringstream &cocciFptr, string arrayPrefix,
        bool a, bool b, bool c) {

    cocciFptr << "+  \n";
    cocciFptr << "+  /* Free device memory */    \n";
    if (a)
        cocciFptr << "+  cudaFree(" << arrayPrefix << "_A); \n";
    if (b)
        cocciFptr << "+  cudaFree(" << arrayPrefix << "_B); \n";
    if (c)
        cocciFptr << "+  cudaFree(" << arrayPrefix << "_C); \n";
    cocciFptr << "+ \n";

}

inline void FreeDeviceMemoryB2(ostringstream &cocciFptr, string arrayPrefix,
        bool a, bool x, bool y) {

    cocciFptr << "+  \n";
    cocciFptr << "+  /* Free device memory */    \n";
    if (a)
        cocciFptr << "+  cudaFree(" << arrayPrefix << "_A); \n";
    if (x)
        cocciFptr << "+  cudaFree(" << arrayPrefix << "_X); \n";
    if (y)
        cocciFptr << "+  cudaFree(" << arrayPrefix << "_Y); \n";
    cocciFptr << "+ \n";

}

inline void DeclareDevicePtrB3(ostringstream &cocciFptr, string aType,
        string arrayPrefix, bool a, bool b, bool c) {
    cocciFptr << "+  \n";
    if (a)
        cocciFptr << "+  " << aType << " *" << arrayPrefix << "_A;  \n";
    if (b)
        cocciFptr << "+  " << aType << " *" << arrayPrefix << "_B;  \n";
    if (c)
        cocciFptr << "+  " << aType << " *" << arrayPrefix << "_C;   \n";
    cocciFptr << "+  int sizeType_" << arrayPrefix << " = sizeof(" << aType
            << "); \n";
    cocciFptr << "+  \n";
}

inline void DeclareDevicePtrB2(ostringstream &cocciFptr, string aType,
        string arrayPrefix, bool a, bool x, bool y) {
    cocciFptr << "+  \n";
    if (a)
        cocciFptr << "+  " << aType << " *" << arrayPrefix << "_A;  \n";
    if (x)
        cocciFptr << "+  " << aType << " *" << arrayPrefix << "_X;  \n";
    if (y)
        cocciFptr << "+  " << aType << " *" << arrayPrefix << "_Y;   \n";
    cocciFptr << "+  int sizeType_" << arrayPrefix << " = sizeof(" << aType
            << "); \n";
    cocciFptr << "+  \n";
}

inline void memAllocCheck(ostringstream &cocciFptr, string array){
    cocciFptr << "+  if(CudaStat != cudaSuccess) {\n";
    cocciFptr
            << "+              printf(\"Error allocating memory on device for array "
            << array << "!!\\n\");\n";
    cocciFptr << "+          return -1;\n";
    cocciFptr << "+  }\n";
    cocciFptr << "+  \n";
}

inline void memCpyCheck(ostringstream &cocciFptr, string array){
    cocciFptr << "+  if(CudaStatReturn != CUBLAS_STATUS_SUCCESS) {\n";
    cocciFptr
            << "+              printf(\"data download/upload failed for array "
            << array << "!!\\n\");\n";
    cocciFptr << "+          return -1;\n";
    cocciFptr << "+  }\n";
    cocciFptr << "+  \n";
}

inline void blasSuccessCheck(ostringstream &cocciFptr, string blasCall){
    cocciFptr << "+  if(CudaStatReturn != CUBLAS_STATUS_SUCCESS) {\n";
    cocciFptr
            << "+              printf(\"" << blasCall << " failed!!\\n\");\n";
    cocciFptr << "+          return -1;\n";
    cocciFptr << "+  }\n";
    cocciFptr << "+  \n";
}

#endif /* BLAS2CUBLAS_H_ */

