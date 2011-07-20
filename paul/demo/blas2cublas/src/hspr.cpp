#include "blas2cublas.h"

using namespace std;

void handleHSPR(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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

	SgNode* matrixAptr = NULL;
	SgNode* vecXptr = NULL;

	if(checkBlasCallType){
		cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
		vecXptr = fArgs->get_traversalSuccessorByIndex(4);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		vecXptr = fArgs->get_traversalSuccessorByIndex(3);

	}

	if(cblasUplo == "CblasUpper") uplo = "\'U\'";
	else if(cblasUplo == "CblasLower") uplo = "\'L\'";

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();

	if(fname.find("chpr") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasChpr";
	}
	else if(fname.find("zhpr") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZhpr";
	}
	else if(fname.find("sspr") != string::npos){
		aType = "float";
		cublasCall = "cublasSspr";
	}
	else if(fname.find("dspr") != string::npos){
		aType = "double";
		cublasCall = "cublasDspr";
	}


	cocciFptr << "@@ \n";
	cocciFptr << "identifier order,uplo;  \n";
	cocciFptr << "expression n, alpha, incx;  \n";
	cocciFptr << "@@ \n";

	if(checkBlasCallType)
		cocciFptr << "- "<<blasCall<<"(order,uplo, n, alpha,"<<vecXRef<<",incx,"<<matARef<<"); \n";
	else cocciFptr << "- "<<blasCall<<"(uplo, n, alpha,"<<vecXRef<<",incx,"<<matARef<<"); \n";

	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,true,true,false);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciFptr << "+  cublasAlloc("<<len_X<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy matrix, vectors to device */     \n";
	cocciFptr << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
	cocciFptr << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";


	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciFptr,warnRowMajor);

	if(uplo==""){
		//cocciFptr << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cocciFptr << "+  "<<cublasCall<<"(uplo,n, alpha,"<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_A);  \n";
	}

	else
		cocciFptr << "+  "<<cublasCall<<"("<<uplo<<",n, alpha,"<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_A);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result matrix back to host */  \n";
	cocciFptr << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<arrayPrefix<<"_A, n, (void *) "<<matARef<<", n);  \n";
	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,true,true,false);

}

