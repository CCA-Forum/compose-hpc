#include "blas2cublas.h"

using namespace std;

void handleHESYR2(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){
	
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
		matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
		vecXptr = fArgs->get_traversalSuccessorByIndex(4);
		vecYptr = fArgs->get_traversalSuccessorByIndex(6);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
		vecXptr = fArgs->get_traversalSuccessorByIndex(3);
		vecYptr = fArgs->get_traversalSuccessorByIndex(5);

	}

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("cher2") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCher2";
	}
	else if(fname.find("zher2") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZher2";
	}
	else if(fname.find("ssyr2") != string::npos){
		aType = "float";
		cublasCall = "cublasSsyr2";
	}
	else if(fname.find("dsyr2") != string::npos){
		aType = "double";
		cublasCall = "cublasDsyr2";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression order,uplo;  \n";
	cocciStream << "expression n, alpha, lda, incx, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<<",lda); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<<",lda); \n";

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,true,true,true);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<uPrefix<<", (void *)"<<matARef<<", n, (void *) "<<uPrefix<<"_A, n);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecXRef<<", incx, "<<uPrefix<<"_X, incx);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecYRef<<", incy, "<<uPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		RowMajorWarning(cocciStream,isRowMajor);

		if(cblasUplo == "CblasUpper") uplo = "\'U\'";
		else if(cblasUplo == "CblasLower") uplo = "\'L\'";
		else{
			uplo = uPrefix + "_uplo";
			cocciStream << "+ char "<<uplo<<"; \n";
			cocciStream << "+ if("<<cblasUplo<<" == CblasUpper) "<<uplo<<" = \'U\'; \n";
			cocciStream << "+ else "<<uplo<<" = \'L\'; \n";

		}

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<",n, alpha,"<<uPrefix<<"_X,incx,"<<uPrefix<<"_Y,incy,"<<uPrefix<<"_A,lda);  \n\n";
		cocciStream << "+  /* Copy result matrix back to host */  \n";
		cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<uPrefix<<", (void *)"<<uPrefix<<"_A, n, (void *) "<<matARef<<", n);  \n";
	}

	else{

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(n), sizeType_"<<uPrefix<<", (void *)"<<matARef<<", *(n), (void *) "<<uPrefix<<"_A, *(n));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecYRef<<", *(incy), "<<uPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(n), *(alpha),"<<uPrefix<<"_X,*(incx),"<<uPrefix<<"_Y,*(incy),"<<uPrefix<<"_A, *(lda));  \n\n";
		cocciStream << "+  /* Copy result matrix back to host */  \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(n), sizeType_"<<uPrefix<<", (void *)"<<uPrefix<<"_A, *(n), (void *) "<<matARef<<", *(n));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,uPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

