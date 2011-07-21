
#ifndef BLAS2CUBLAS_H_
#define BLAS2CUBLAS_H_

#include <iostream>
#include <fstream>
#include "rose.h"
using namespace std;

void handleGEMM(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleSYHEMM(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleSYHERK(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleSYHER2K(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleTRSMM(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleGBMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleGEMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleGER(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleHSBMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleHSEYMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleHESYR2(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleHESYR(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleHSPMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleHSPR2(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleHSPR(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleTBSMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleTPSMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleTRSMV(ofstream &, bool, bool, string, string, SgExprListExp*);
void handleSumNrm2Aminmax(ofstream &, string, string, SgExprListExp*);
void handleAXPY(ofstream &, string, string, SgExprListExp*);
void handleAXPBY(ofstream &, string, string, SgExprListExp*);
void handleCOPY(ofstream &, string, string, SgExprListExp*);
void handleDOT(ofstream &, string, string, SgExprListExp*);
void handleSCAL(ofstream &, string, string, SgExprListExp*);
void handleSWAP(ofstream &, string, string, SgExprListExp*);
void handleROTM(ofstream &, string, string, SgExprListExp*);
void handleROT(ofstream &, string, string, SgExprListExp*);

inline void RowMajorWarning(ofstream &cocciFptr, bool warnRowMajor){
	if(warnRowMajor){
		cocciFptr << "+  //BLAS_TO_CUBLAS transformation performance warning: \n";
		cocciFptr << "+  //CUBLAS calls assume arrays are stored in column-major order. \n";
		cocciFptr << "+  //The original BLAS call specified that the arrays are stored in row-major order. \n";
	}
}

inline void FreeDeviceMemoryB3(ofstream &cocciFptr, string arrayPrefix, bool a, bool b, bool c){

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Free device memory */    \n";
	if(a) cocciFptr << "+  cublasFree("<<arrayPrefix<<"_A); \n";
	if(b) cocciFptr << "+  cublasFree("<<arrayPrefix<<"_B); \n";
	if(c) cocciFptr << "+  cublasFree("<<arrayPrefix<<"_C); \n";
	cocciFptr << "+ \n";

}

inline void FreeDeviceMemoryB2(ofstream &cocciFptr, string arrayPrefix, bool a, bool x, bool y){

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Free device memory */    \n";
	if(a) cocciFptr << "+  cublasFree("<<arrayPrefix<<"_A); \n";
	if(x) cocciFptr << "+  cublasFree("<<arrayPrefix<<"_X); \n";
	if(y) cocciFptr << "+  cublasFree("<<arrayPrefix<<"_Y); \n";
	cocciFptr << "+ \n";

}

inline void DeclareDevicePtrB3(ofstream &cocciFptr, string aType, string arrayPrefix, bool a, bool b, bool c){
	cocciFptr << "+  \n";
	if(a) cocciFptr << "+  "<<aType<<" *"<<arrayPrefix<<"_A;  \n";
	if(b) cocciFptr << "+  "<<aType<<" *"<<arrayPrefix<<"_B;  \n";
	if(c) cocciFptr << "+  "<<aType<<" *"<<arrayPrefix<<"_C;   \n";
	cocciFptr << "+  int sizeType_"<<arrayPrefix<<" = sizeof("<<aType<<"); \n";
	cocciFptr << "+  \n";
}

inline void DeclareDevicePtrB2(ofstream &cocciFptr, string aType, string arrayPrefix, bool a, bool x, bool y){
	cocciFptr << "+  \n";
	if(a) cocciFptr << "+  "<<aType<<" *"<<arrayPrefix<<"_A;  \n";
	if(x) cocciFptr << "+  "<<aType<<" *"<<arrayPrefix<<"_X;  \n";
	if(y) cocciFptr << "+  "<<aType<<" *"<<arrayPrefix<<"_Y;   \n";
	cocciFptr << "+  int sizeType_"<<arrayPrefix<<" = sizeof("<<aType<<"); \n";
	cocciFptr << "+  \n";
}

#endif /* BLAS2CUBLAS_H_ */

