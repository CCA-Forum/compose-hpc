#include "blas2cublas.h"

using namespace std;

void handleTBSMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo,trans,diag;  \n";
	cocciStream << "expression n, k, lda, incx;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, trans,diag, n, k, "<<matARef<<",lda,"<<vecXRef<<",incx); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, trans,diag, n, k,"<<matARef<<",lda,"<<vecXRef<<",incx); \n";

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

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<","<<cbTrans<<","<<diag<<",n,k,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx);  \n\n";
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

		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(trans),*(diag),*(n),*(k),"<<arrayPrefix<<"_A, *(lda),"<<arrayPrefix<<"_X,*(incx));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
	}


	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

