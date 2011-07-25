#include "blas2cublas.h"

using namespace std;

void handleHSBMV(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;
	string prefix = "";
	string len_X = "n";
	string len_Y = "n";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	len_Y = arrayPrefix.substr(lenInd+1);

	arrayPrefix = prefix;

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

	if(cblasUplo == "CblasUpper") uplo = "\'U\'";
	else if(cblasUplo == "CblasLower") uplo = "\'L\'";

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

	cocciStream << "@@ \n";
	cocciStream << "identifier order,uplo;  \n";
	cocciStream << "expression n, k, alpha, lda, incx, beta, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, n, k, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, n, k, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,true);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n*k, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc("<<len_X<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc("<<len_Y<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetMatrix ( n,k, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
	cocciStream << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  if(beta != 0) cublasSetVector ( len_Y, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);

	if(uplo==""){
		//cocciStream << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cocciStream << "+  "<<cublasCall<<"(uplo,n,k, alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n";
	}

	else
		cocciStream << "+  "<<cublasCall<<"("<<uplo<<",n,k,  alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector ( len_Y, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

