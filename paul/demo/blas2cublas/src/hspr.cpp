#include "blas2cublas.h"

using namespace std;

void handleHSPR(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;


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


	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo;  \n";
	cocciStream << "expression n, alpha, incx;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, n, alpha,"<<vecXRef<<",incx,"<<matARef<<"); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, n, alpha,"<<vecXRef<<",incx,"<<matARef<<"); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,false);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		RowMajorWarning(cocciStream,isRowMajor);

		if(cblasUplo == "CblasUpper") uplo = "\'U\'";
		else if(cblasUplo == "CblasLower") uplo = "\'L\'";

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<",n, alpha,"<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_A);  \n\n";
		cocciStream << "+  /* Copy result matrix back to host */  \n";
		cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<arrayPrefix<<"_A, n, (void *) "<<matARef<<", n);  \n";
	}

	else{

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n\n";

		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(n), sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", *(n), (void *) "<<arrayPrefix<<"_A, *(n));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(uplo), *(n), *(alpha),"<<arrayPrefix<<"_X,*(incx),"<<arrayPrefix<<"_A);  \n\n";
		cocciStream << "+  /* Copy result matrix back to host */  \n";
		cocciStream << "+  cublasSetMatrix ( *(n), *(n), sizeType_"<<arrayPrefix<<", (void *)"<<arrayPrefix<<"_A, *(n), (void *) "<<matARef<<", *(n));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

