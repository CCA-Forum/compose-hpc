#include "blas2cublas.h"

using namespace std;

void handleSWAP(ofstream &cocciFptr, bool checkBlasCallType, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(3);

	string vecXRef = vecXptr->unparseToCompleteString();
	string vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("sswap") != string::npos){
		aType = "float";
		cublasCall = "cublasSswap";
	}
	else if(fname.find("dswap") != string::npos){
		aType = "double";
		cublasCall = "cublasDswap";
	}
	else if(fname.find("cswap") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCswap";
	}
	else if(fname.find("zswap") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZswap";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression n, incx, incy;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n, "<<vecXRef<<",incx,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,true);

	if(checkBlasCallType){
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";

		cocciStream << "+  /* Copy vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy);  \n\n";

		cocciStream << "+  /* Copy swapped vectors back to host */  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else{
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";

		cocciStream << "+  /* Copy vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecYRef<<", *(incy), "<<arrayPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(n), "<<arrayPrefix<<"_X,*(incx),"<<arrayPrefix<<"_Y,*(incy));  \n\n";

		cocciStream << "+  /* Copy swapped vectors back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,true);
	cocciFptr << cocciStream.str();

}

