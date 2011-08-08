#include "blas2cublas.h"

using namespace std;

void handleHSBMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

	string matARef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	string cbTrans="";
	string cblasUplo = "";
	string uplo = "";

	string vecXRef="";
	string vecYRef="";

	SgNode* matrixAptr = NULL;
	SgNode* vecXptr = NULL;
	SgNode* vecYptr = NULL;

	if(checkBlasCallType){
		cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		vecXptr = fArgs->get_traversalSuccessorByIndex(7);
		vecYptr = fArgs->get_traversalSuccessorByIndex(10);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
		vecXptr = fArgs->get_traversalSuccessorByIndex(6);
		vecYptr = fArgs->get_traversalSuccessorByIndex(9);

	}

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("chbmv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasChbmv";
	}
	else if(fname.find("zhbmv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZhbmv";
	}
	else if(fname.find("ssbmv") != string::npos){
		aType = "float";
		cublasCall = "cublasSsbmv";
	}
	else if(fname.find("dsbmv") != string::npos){
		aType = "double";
		cublasCall = "cublasDsbmv";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo;  \n";
	cocciStream << "expression n, k, alpha, lda, incx, beta, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, n, k, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, n, k, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,true);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n*k, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( n,k, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
		cocciStream << "+  if(beta != 0) cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		RowMajorWarning(cocciStream,isRowMajor);

		if(cblasUplo == "CblasUpper") uplo = "\'U\'";
		else if(cblasUplo == "CblasLower") uplo = "\'L\'";

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<",n,k,  alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else {

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(k), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(k), sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", *(n), (void *) "<<arrayPrefix<<"_A, *(n));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  if(*(beta) != 0) cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecYRef<<", *(incy), "<<arrayPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(n),*(k),  *(alpha),"<<arrayPrefix<<"_A, *(lda),"<<arrayPrefix<<"_X,*(incx),*(beta),"<<arrayPrefix<<"_Y,*(incy));  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}	

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

