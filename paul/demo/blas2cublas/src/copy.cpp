#include "blas2cublas.h"

using namespace std;

void handleCOPY(ofstream &cocciFptr, bool checkBlasCallType, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

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

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression n, incx, incy;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n, "<<vecXRef<<",incx,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,true);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy);  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else{

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(n), "<<arrayPrefix<<"_X,*(incx),"<<arrayPrefix<<"_Y,*(incy));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,true);
	cocciFptr << cocciStream.str();

}

