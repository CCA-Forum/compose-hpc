#include "blas2cublas.h"

using namespace std;

void handleROT(ofstream &cocciFptr, bool checkBlasCallType, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(3);

	string vecXRef = vecXptr->unparseToCompleteString();
	string vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("srot") != string::npos){
		aType = "float";
		cublasCall = "cublasSrot";
	}
	else if(fname.find("drot") != string::npos){
		aType = "double";
		cublasCall = "cublasDrot";
	}
	else if(fname.find("crot") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCrot";
	}
	else if(fname.find("zrot") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZrot";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression n, incx, incy, cRef, sRef;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n,"<<vecXRef<<",incx,"<<vecYRef<<",incy, cRef, sRef); \n";

	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_result;  \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,true);

	if(checkBlasCallType){
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
		cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_result);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy, cRef, sRef);  \n\n";

		cocciStream << "+  /* Copy result vectors back to host */  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else{
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
		cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_result);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecYRef<<", *(incy), "<<arrayPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(n), "<<arrayPrefix<<"_X,*(incx),"<<arrayPrefix<<"_Y,*(incy), *(cRef), *(sRef));  \n\n";

		cocciStream << "+  /* Copy result vectors back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,true);
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_result); \n";
	cocciFptr << cocciStream.str();

}

