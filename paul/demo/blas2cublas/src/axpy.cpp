#include "blas2cublas.h"

using namespace std;

void handleAXPY(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;
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

	SgNode* matrixAptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(2);
	SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(4);

	string matARef = matrixAptr->unparseToCompleteString();
	string vecXRef = vecXptr->unparseToCompleteString();
	string vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("saxpy") != string::npos){
		aType = "float";
		cublasCall = "cublasSaxpy";
	}
	else if(fname.find("daxpy") != string::npos){
		aType = "double";
		cublasCall = "cublasDaxpy";
	}
	else if(fname.find("caxpy") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCaxpy";
	}
	else if(fname.find("zaxpy") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZaxpy";
	}

	cocciStream << "@@ \n";
	cocciStream << "expression n, incx, incy;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n, "<<matARef<<","<<vecXRef<<",incx,"<<vecYRef<<",incy); \n";

	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_A;  \n";
	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,true);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";
	cocciStream << "+  cudaMemcpy("<<arrayPrefix<<"_A,"<<matARef<<",sizeType_"<<arrayPrefix<<",cudaMemcpyHostToDevice);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	cocciStream << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_A, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,true);
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_A); \n";
	cocciFptr << cocciStream.str();

}

