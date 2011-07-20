#include "blas2cublas.h"

using namespace std;

void handleDOT(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	string prefix = "";
	string len_X = "";
	string len_Y = "";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	len_Y = arrayPrefix.substr(lenInd+1);

	arrayPrefix = prefix;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(3);

	string vecXRef = vecXptr->unparseToCompleteString();
	string vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("cdotc") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCdotc";
	}
	else if(fname.find("zdotc") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZdotc";
	}
	else if(fname.find("cdotu") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCdotu";
	}
	else if(fname.find("zdotu") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZdotu";
	}
	else if(fname.find("sdot") != string::npos){
		aType = "float";
		cublasCall = "cublasSdot";
	}
	else if(fname.find("ddot") != string::npos){
		aType = "double";
		cublasCall = "cublasDdot";
	}

	cocciFptr << "@@ \n";
	cocciFptr << "expression n, incx, incy;  \n";
	cocciFptr << "@@ \n";

	cocciFptr << "<...\n- "<<blasCall<<"(n, "<<vecXRef<<",incx,"<<vecYRef<<",incy); \n";
	cocciFptr << "+ "<<aType<<" *"<<arrayPrefix<<"_result;  \n";

	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,false,true,true);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciFptr << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciFptr << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_result);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy matrix, vectors to device */     \n";
	cocciFptr << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciFptr << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";
	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	cocciFptr << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy);  \n...>\n";
	cocciFptr << "+  \n";
	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,false,true,true);
	cocciFptr << "+  cublasFree("<<arrayPrefix<<"_result); \n";

}
