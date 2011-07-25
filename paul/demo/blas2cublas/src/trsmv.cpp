#include "blas2cublas.h"

using namespace std;

void handleTRSMV(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;
	string prefix = "";
	string len_X = "n";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	arrayPrefix = prefix;

	string matARef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	string cblasUplo = "";
	string uplo = "";
	string cblasTrans = "";
	string cbTrans = "";
	string cblasDiag = "";
	string diag = "";
	string vecXRef="";

	SgNode* matrixAptr = NULL;
	SgNode* vecXptr = NULL;

	if(checkBlasCallType){
		cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
		cblasDiag = fArgs->get_traversalSuccessorByIndex(3)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		vecXptr = fArgs->get_traversalSuccessorByIndex(7);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasDiag = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
		vecXptr = fArgs->get_traversalSuccessorByIndex(6);
	}


	if(cblasUplo == "CblasUpper") uplo = "\'U\'";
	else if(cblasUplo == "CblasLower") uplo = "\'L\'";

	if(    cblasTrans  == "CblasTrans")     cbTrans = "\'T\'";
	else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'N\'";
	else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";

	if(cblasDiag == "CblasNonUnit") diag = "\'N\'";
	else if(cblasDiag == "CblasUnit") diag = "\'U\'";

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();

	if(fname.find("ctrmv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtrmv";
	}
	else if(fname.find("ztrmv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtrmv";
	}
	else if(fname.find("strmv") != string::npos){
		aType = "float";
		cublasCall = "cublasStrmv";
	}
	else if(fname.find("dtrmv") != string::npos){
		aType = "double";
		cublasCall = "cublasDtrmv";
	}
	else if(fname.find("ctrsv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtrsv";
	}
	else if(fname.find("ztrsv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtrsv";
	}
	else if(fname.find("strsv") != string::npos){
		aType = "float";
		cublasCall = "cublasStrsv";
	}
	else if(fname.find("dtrsv") != string::npos){
		aType = "double";
		cublasCall = "cublasDtrsv";
	}

	cocciStream << "@@ \n";
	cocciStream << "identifier order,uplo,trans,diag;  \n";
	cocciStream << "expression n, k, lda, incx;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, trans,diag, n, "<<matARef<<",lda,"<<vecXRef<<",incx); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, trans,diag, n, "<<matARef<<",lda,"<<vecXRef<<",incx); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,false);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc("<<len_X<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vector to device */     \n";
	cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
	cocciStream << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);

	if(uplo==""){
		//cocciStream << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		uplo = "uplo";
	}

	if(cblasTrans == ""){
		cbTrans = "trans";
		//cocciStream << "//Warning:CBLAS_TRANS could not be determined. Default = \'N\' \n";
	}

	if(cblasDiag == ""){
		diag = "diag";
		//cocciStream << "//Warning:CBLAS_DIAG could not be determined. Default = \'N\' \n";
	}


	cocciStream << "+  "<<cublasCall<<"("<<uplo<<","<<cbTrans<<","<<diag<<",n,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

