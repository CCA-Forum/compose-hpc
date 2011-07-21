#include "blas2cublas.h"

using namespace std;

void handleTPSMV(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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
		vecXptr = fArgs->get_traversalSuccessorByIndex(6);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasDiag = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
		vecXptr = fArgs->get_traversalSuccessorByIndex(5);
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

	if(fname.find("ctpmv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtpmv";
	}
	else if(fname.find("ztpmv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtpmv";
	}
	else if(fname.find("stpmv") != string::npos){
		aType = "float";
		cublasCall = "cublasStpmv";
	}
	else if(fname.find("dtpmv") != string::npos){
		aType = "double";
		cublasCall = "cublasDtpmv";
	}
	else if(fname.find("ctpsv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtpsv";
	}
	else if(fname.find("ztpsv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtpsv";
	}
	else if(fname.find("stpsv") != string::npos){
		aType = "float";
		cublasCall = "cublasStpsv";
	}
	else if(fname.find("dtpsv") != string::npos){
		aType = "double";
		cublasCall = "cublasDtpsv";
	}

	cocciFptr << "@@ \n";
	cocciFptr << "identifier order,uplo,trans,diag;  \n";
	cocciFptr << "expression n, incx;  \n";
	cocciFptr << "@@ \n";

	if(checkBlasCallType)
		cocciFptr << "- "<<blasCall<<"(order,uplo, trans,diag, n, "<<matARef<<","<<vecXRef<<",incx); \n";
	else cocciFptr << "- "<<blasCall<<"(uplo, trans,diag, n, "<<matARef<<","<<vecXRef<<",incx); \n";

	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,true,true,false);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciFptr << "+  cublasAlloc("<<len_X<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy matrix, vector to device */     \n";
	cocciFptr << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
	cocciFptr << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciFptr,warnRowMajor);

	if(uplo==""){
		//cocciFptr << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		uplo = "uplo";
	}

	if(cblasTrans == ""){
		cbTrans = "trans";
		//cocciFptr << "//Warning:CBLAS_TRANS could not be determined. Default = \'N\' \n";
	}

	if(cblasDiag == ""){
		diag = "diag";
		//cocciFptr << "//Warning:CBLAS_DIAG could not be determined. Default = \'N\' \n";
	}


	cocciFptr << "+  "<<cublasCall<<"("<<uplo<<","<<cbTrans<<","<<diag<<",n,"<<arrayPrefix<<"_A,"<<arrayPrefix<<"_X,incx);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result vector back to host */  \n";
	cocciFptr << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,true,true,false);

}

