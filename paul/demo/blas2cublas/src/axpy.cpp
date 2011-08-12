#include "blas2cublas.h"

using namespace std;

void handleAXPY(ofstream &cocciFptr, bool checkBlasCallType, string fname, string uPrefix, SgExprListExp* fArgs){
	
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

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,false,true,true);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecXRef<<", incx, "<<uPrefix<<"_X, incx);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecYRef<<", incy, "<<uPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(n, alpha, "<<uPrefix<<"_X,incx,"<<uPrefix<<"_Y,incy);  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else{
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecYRef<<", *(incy), "<<uPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(n), *(alpha), "<<uPrefix<<"_X,*(incx),"<<uPrefix<<"_Y,*(incy));  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,uPrefix,false,true,true);
	cocciFptr << cocciStream.str();

}

