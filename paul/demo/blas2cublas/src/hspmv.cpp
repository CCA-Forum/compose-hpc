#include "blas2cublas.h"

using namespace std;

void handleHSPMV(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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
		matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
		vecXptr = fArgs->get_traversalSuccessorByIndex(5);
		vecYptr = fArgs->get_traversalSuccessorByIndex(8);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(3);
		vecXptr = fArgs->get_traversalSuccessorByIndex(4);
		vecYptr = fArgs->get_traversalSuccessorByIndex(7);

	}

	if(cblasUplo == "CblasUpper") uplo = "\'U\'";
	else if(cblasUplo == "CblasLower") uplo = "\'L\'";

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("chpmv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasChpmv";
	}
	else if(fname.find("zhpmv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZhpmv";
	}
	else if(fname.find("sspmv") != string::npos){
		aType = "float";
		cublasCall = "cublasSspmv";
	}
	else if(fname.find("dspmv") != string::npos){
		aType = "double";
		cublasCall = "cublasDspmv";
	}

	cocciFptr << "@@ \n";
	cocciFptr << "identifier order,uplo;  \n";
	cocciFptr << "expression n, alpha, incx, beta, incy;  \n";
	cocciFptr << "@@ \n";

	if(checkBlasCallType)
		cocciFptr << "- "<<blasCall<<"(order,uplo, n, alpha,"<<matARef<<","<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciFptr << "- "<<blasCall<<"(uplo, n, alpha,"<<matARef<<","<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,true,true,true);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciFptr << "+  cublasAlloc("<<len_X<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciFptr << "+  cublasAlloc("<<len_Y<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy matrix, vectors to device */     \n";
	cocciFptr << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
	cocciFptr << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciFptr << "+  if(beta != 0) cublasSetVector ( len_Y, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciFptr,warnRowMajor);

	if(uplo==""){
		//cocciFptr << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cocciFptr << "+  "<<cublasCall<<"(uplo,n, alpha,"<<arrayPrefix<<"_A,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n";
	}

	else
		cocciFptr << "+  "<<cublasCall<<"("<<uplo<<",n, alpha,"<<arrayPrefix<<"_A,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result vector back to host */  \n";
	cocciFptr << "+  cublasSetVector ( len_Y, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,true,true,true);

}

