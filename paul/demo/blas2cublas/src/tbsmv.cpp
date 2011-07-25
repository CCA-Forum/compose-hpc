#include "blas2cublas.h"

using namespace std;

void handleTBSMV(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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
		matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
		vecXptr = fArgs->get_traversalSuccessorByIndex(8);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasDiag = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		vecXptr = fArgs->get_traversalSuccessorByIndex(7);
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

	if(fname.find("ctbmv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtbmv";
	}
	else if(fname.find("ztbmv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtbmv";
	}
	else if(fname.find("stbmv") != string::npos){
		aType = "float";
		cublasCall = "cublasStbmv";
	}
	else if(fname.find("dtbmv") != string::npos){
		aType = "double";
		cublasCall = "cublasDtbmv";
	}
	else if(fname.find("ctbsv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtbsv";
	}
	else if(fname.find("ztbsv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtbsv";
	}
	else if(fname.find("stbsv") != string::npos){
		aType = "float";
		cublasCall = "cublasStbsv";
	}
	else if(fname.find("dtbsv") != string::npos){
		aType = "double";
		cublasCall = "cublasDtbsv";
	}

	cocciStream << "@@ \n";
	cocciStream << "identifier order,uplo,trans,diag;  \n";
	cocciStream << "expression n, k, lda, incx;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, trans,diag, n, k, "<<matARef<<",lda,"<<vecXRef<<",incx); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, trans,diag, n, k,"<<matARef<<",lda,"<<vecXRef<<",incx); \n";

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


	cocciStream << "+  "<<cublasCall<<"("<<uplo<<","<<cbTrans<<","<<diag<<",n,k,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

