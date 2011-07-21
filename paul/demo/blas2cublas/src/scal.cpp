#include "blas2cublas.h"

using namespace std;

void handleSCAL(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	string prefix = "";
	string len_X = "";
	string len_Y = "";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	arrayPrefix = prefix;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	SgNode* matrixAptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(2);

	string matARef = matrixAptr->unparseToCompleteString();
	string vecXRef = vecXptr->unparseToCompleteString();

	if(fname.find("sscal") != string::npos){
		aType = "float";
		cublasCall = "cublasSscal";
	}
	else if(fname.find("dscal") != string::npos){
		aType = "double";
		cublasCall = "cublasDscal";
	}
	else if(fname.find("cscal") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCscal";
	}
	else if(fname.find("zscal") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZscal";
	}
	else if(fname.find("csscal") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCsscal";
	}
	else if(fname.find("zdscal") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZdscal";
	}

	cocciFptr << "@@ \n";
	cocciFptr << "expression n, incx;  \n";
	cocciFptr << "@@ \n";

	cocciFptr << "- "<<blasCall<<"(n, "<<matARef<<","<<vecXRef<<",incx); \n";

	cocciFptr << "+ "<<aType<<" *"<<arrayPrefix<<"_A;  \n";
	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,false,true,false);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciFptr << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy matrix, vectors to device */     \n";
	cocciFptr << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciFptr << "+  cudaMemcpy("<<arrayPrefix<<"_A,"<<matARef<<",sizeType_"<<arrayPrefix<<",cudaMemcpyHostToDevice);  \n";


	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	cocciFptr << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_A, "<<arrayPrefix<<"_X,incx);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result vector back to host */  \n";
	cocciFptr << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,false,true,false);
	cocciFptr << "+  cublasFree("<<arrayPrefix<<"_A); \n";
}

