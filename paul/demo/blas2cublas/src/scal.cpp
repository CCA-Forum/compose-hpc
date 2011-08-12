#include "blas2cublas.h"

using namespace std;

void handleSCAL(ofstream &cocciFptr, bool checkBlasCallType, string fname, string uPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";


	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(2);
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

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression n, a, incx;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n, a,"<<vecXRef<<",incx); \n";

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,false,true,false);

	if(checkBlasCallType){
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecXRef<<", incx, "<<uPrefix<<"_X, incx);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(n, a, "<<uPrefix<<"_X,incx);  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<uPrefix<<","<<uPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	}

	else{
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(n), *(a), "<<uPrefix<<"_X,*(incx));  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<uPrefix<<","<<uPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,uPrefix,false,true,false);
	cocciFptr << cocciStream.str();
}

