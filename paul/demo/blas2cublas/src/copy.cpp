#include "blas2cublas.h"

using namespace std;

void handleCOPY(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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

	if(fname.find("scopy") != string::npos){
		aType = "float";
		cublasCall = "cublasScopy";
	}
	else if(fname.find("dcopy") != string::npos){
		aType = "double";
		cublasCall = "cublasDcopy";
	}
	else if(fname.find("ccopy") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCcopy";
	}
	else if(fname.find("zcopy") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZcopy";
	}

	cocciFptr << "@@ \n";
	cocciFptr << "expression n, incx, incy;  \n";
	cocciFptr << "@@ \n";

	cocciFptr << "- "<<blasCall<<"(n, "<<vecXRef<<",incx,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,false,true,true);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciFptr << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy matrix, vectors to device */     \n";
	cocciFptr << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	cocciFptr << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result vector back to host */  \n";
	cocciFptr << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,false,true,true);

}

