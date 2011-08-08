#include "blas2cublas.h"

using namespace std;

void handleTRSMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;

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

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo,trans,diag;  \n";
	cocciStream << "expression n, k, lda, incx;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, trans,diag, n, "<<matARef<<",lda,"<<vecXRef<<",incx); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, trans,diag, n, "<<matARef<<",lda,"<<vecXRef<<",incx); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,false);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n\n";
		cocciStream << "+  /* Copy matrix, vector to device */     \n";
		cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", n, (void *) "<<arrayPrefix<<"_A, n);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";

		if(cblasUplo == "CblasUpper") uplo = "\'U\'";
		else if(cblasUplo == "CblasLower") uplo = "\'L\'";

		if(isRowMajor){
			if(    cblasTrans  == "CblasTrans")     cbTrans = "\'N\'";
			else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'T\'";
			else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";
		}
		else{
			if(    cblasTrans  == "CblasTrans")     cbTrans = "\'T\'";
			else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'N\'";
			else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";
		}

		if(cblasDiag == "CblasNonUnit") diag = "\'N\'";
		else if(cblasDiag == "CblasUnit") diag = "\'U\'";

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<","<<cbTrans<<","<<diag<<",n,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx);  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	}

	else{

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n\n";
		cocciStream << "+  /* Copy matrix, vector to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(n), sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", *(n), (void *) "<<arrayPrefix<<"_A, *(n));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";

		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(trans),*(diag),*(n),"<<arrayPrefix<<"_A, *(lda),"<<arrayPrefix<<"_X,*(incx));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

