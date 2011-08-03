#include "blas2cublas.h"

using namespace std;

void handleAXPY(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(2);
	SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(4);

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

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression n, alpha, incx, incy;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n, alpha, "<<vecXRef<<",incx,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,true);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	cocciStream << "+  "<<cublasCall<<"(n, alpha, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,true);
	cocciFptr << cocciStream.str();

}

