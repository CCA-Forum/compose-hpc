#include "blas2cublas.h"

using namespace std;

void handleHSEYMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){
	
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
		matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
		vecXptr = fArgs->get_traversalSuccessorByIndex(6);
		vecYptr = fArgs->get_traversalSuccessorByIndex(9);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(3);
		vecXptr = fArgs->get_traversalSuccessorByIndex(5);
		vecYptr = fArgs->get_traversalSuccessorByIndex(8);

	}

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("chemv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasChemv";
	}
	else if(fname.find("zhemv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZhemv";
	}
	else if(fname.find("ssymv") != string::npos){
		aType = "float";
		cublasCall = "cublasSsymv";
	}
	else if(fname.find("dsymv") != string::npos){
		aType = "double";
		cublasCall = "cublasDsymv";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression order,uplo;  \n";
	cocciStream << "expression n, alpha, lda, incx, beta, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, n, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, n, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,true,true,true);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<uPrefix<<", (void *)"<<matARef<<", n, (void *) "<<uPrefix<<"_A, n);  \n";
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

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<",n, alpha,"<<uPrefix<<"_A,lda,"<<uPrefix<<"_X,incx,beta,"<<uPrefix<<"_Y,incy);  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	}

	else {

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(n), sizeType_"<<uPrefix<<", (void *)"<<matARef<<", *(n), (void *) "<<uPrefix<<"_A, *(n));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  if(*(beta) != 0) cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecYRef<<", *(incy), "<<uPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(n), *(alpha),"<<uPrefix<<"_A, *(lda),"<<uPrefix<<"_X,*(incx),*(beta),"<<uPrefix<<"_Y,*(incy));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,uPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

