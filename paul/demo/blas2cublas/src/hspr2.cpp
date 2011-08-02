#include "blas2cublas.h"

using namespace std;

void handleHSPR2(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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

	if(cblasUplo == "CblasUpper") uplo = "\'U\'";
	else if(cblasUplo == "CblasLower") uplo = "\'L\'";

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("chpr2") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasChpr2";
	}
	else if(fname.find("zhpr2") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZhpr2";
	}
	else if(fname.find("sspr2") != string::npos){
		aType = "float";
		cublasCall = "cublasSspr2";
	}
	else if(fname.find("dspr2") != string::npos){
		aType = "double";
		cublasCall = "cublasDspr2";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo;  \n";
	cocciStream << "expression n, alpha, incx, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<<"); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<<"); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,true);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);

	if(uplo==""){
		//cocciStream << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cocciStream << "+  "<<cublasCall<<"(uplo,n, alpha,"<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy,"<<arrayPrefix<<"_A);  \n";
	}

	else
		cocciStream << "+  "<<cublasCall<<"("<<uplo<<",n, alpha,"<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy,"<<arrayPrefix<<"_A);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result matrix back to host */  \n";
	cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<arrayPrefix<<"_A, n, (void *) "<<matARef<<", n);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

