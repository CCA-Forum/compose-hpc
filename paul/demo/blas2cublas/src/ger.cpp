#include "blas2cublas.h"

using namespace std;

void handleGER(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

	string matARef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	string vecXRef="";
	string vecYRef="";

	SgNode* matrixAptr = NULL;
	SgNode* vecXptr = NULL;
	SgNode* vecYptr = NULL;

	if(checkBlasCallType){
		matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
		vecXptr = fArgs->get_traversalSuccessorByIndex(4);
		vecYptr = fArgs->get_traversalSuccessorByIndex(6);
	}

	else {
		matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
		vecXptr = fArgs->get_traversalSuccessorByIndex(3);
		vecYptr = fArgs->get_traversalSuccessorByIndex(5);

	}

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("cgerc") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCgerc";
	}
	else if(fname.find("zgerc") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZgerc";
	}
	else if(fname.find("cgeru") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCgeru";
	}
	else if(fname.find("zgeru") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZgeru";
	}
	else if(fname.find("sger") != string::npos){
		aType = "float";
		cublasCall = "cublasSger";
	}
	else if(fname.find("dger") != string::npos){
		aType = "double";
		cublasCall = "cublasDger";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order;  \n";
	cocciStream << "expression m, n, alpha, incx, incy, lda;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,m, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<< ",lda); \n";
	else cocciStream << "- "<<blasCall<<"(m, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<< ",lda); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,true);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(m, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", m, (void *) "<<arrayPrefix<<"_A, m);  \n";
		cocciStream << "+  cublasSetVector ( m, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		RowMajorWarning(cocciStream,isRowMajor);

		cocciStream << "+  "<<cublasCall<<"(m, n, alpha,"<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy,"<<arrayPrefix<<"_A,lda);  \n\n";
		cocciStream << "+  /* Copy result matrix back to host */  \n";
		cocciStream << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<arrayPrefix<<"_A, m, (void *) "<<matARef<<", m);  \n";
	}

	else{

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(m), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", *(m), (void *) "<<arrayPrefix<<"_A, *(m));  \n";
		cocciStream << "+  cublasSetVector ( *(m), sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecYRef<<", *(incy), "<<arrayPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(m), *(n), *(alpha),"<<arrayPrefix<<"_X,*(incx),"<<arrayPrefix<<"_Y,*(incy),"<<arrayPrefix<<"_A,*(lda));  \n\n";
		cocciStream << "+  /* Copy result matrix back to host */  \n";
		cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_"<<arrayPrefix<<", (void *)"<<arrayPrefix<<"_A, *(m), (void *) "<<matARef<<", *(m));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

