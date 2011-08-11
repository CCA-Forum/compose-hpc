#include "blas2cublas.h"

using namespace std;

void handleROT(ofstream &cocciFptr, bool checkBlasCallType, string fname, string uPrefix, SgExprListExp* fArgs){
	
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

	cocciStream << "+ "<<aType<<" *"<<uPrefix<<"_result;  \n";

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,false,true,true);

	if(checkBlasCallType){
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n";
		cocciStream << "+  cublasAlloc(1, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_result);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecXRef<<", incx, "<<uPrefix<<"_X, incx);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecYRef<<", incy, "<<uPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(n, "<<uPrefix<<"_X,incx,"<<uPrefix<<"_Y,incy, cRef, sRef);  \n\n";

		cocciStream << "+  /* Copy result vectors back to host */  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<uPrefix<<","<<uPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
		cocciStream << "+  cublasSetVector (n, sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else{
		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n";
		cocciStream << "+  cublasAlloc(1, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_result);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecYRef<<", *(incy), "<<uPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(n), "<<uPrefix<<"_X,*(incx),"<<uPrefix<<"_Y,*(incy), *(cRef), *(sRef));  \n\n";

		cocciStream << "+  /* Copy result vectors back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<uPrefix<<","<<uPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,uPrefix,false,true,true);
	cocciStream << "+  cublasFree("<<uPrefix<<"_result); \n";
	cocciFptr << cocciStream.str();

}

