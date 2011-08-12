#include "blas2cublas.h"

using namespace std;

void handleHSBMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){
	
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
	cocciStream << "expression order,uplo;  \n";
	cocciStream << "expression n, k, alpha, lda, incx, beta, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, n, k, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, n, k, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,true,true,true);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n*k, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( n,k, sizeType_"<<uPrefix<<", (void *)"<<matARef<<", n, (void *) "<<uPrefix<<"_A, n);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecXRef<<", incx, "<<uPrefix<<"_X, incx);  \n";
		cocciStream << "+  if(beta != 0) cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecYRef<<", incy, "<<uPrefix<<"_Y, incy);  \n\n";

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

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<",n,k,  alpha,"<<uPrefix<<"_A,lda,"<<uPrefix<<"_X,incx,beta,"<<uPrefix<<"_Y,incy);  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else {

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(k), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(k), sizeType_"<<uPrefix<<", (void *)"<<matARef<<", *(n), (void *) "<<uPrefix<<"_A, *(n));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  if(*(beta) != 0) cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecYRef<<", *(incy), "<<uPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(n),*(k),  *(alpha),"<<uPrefix<<"_A, *(lda),"<<uPrefix<<"_X,*(incx),*(beta),"<<uPrefix<<"_Y,*(incy));  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}	

	FreeDeviceMemoryB2(cocciStream,uPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

