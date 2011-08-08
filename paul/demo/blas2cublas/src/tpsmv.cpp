#include "blas2cublas.h"

using namespace std;

void handleTPSMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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
		vecXptr = fArgs->get_traversalSuccessorByIndex(6);
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasDiag = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
		vecXptr = fArgs->get_traversalSuccessorByIndex(5);
	}

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

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo,trans,diag;  \n";
	cocciStream << "expression n, incx;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, trans,diag, n, "<<matARef<<","<<vecXRef<<",incx); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, trans,diag, n, "<<matARef<<","<<vecXRef<<",incx); \n";

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

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<","<<cbTrans<<","<<diag<<",n,"<<arrayPrefix<<"_A,"<<arrayPrefix<<"_X,incx);  \n\n";
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

		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(trans),*(diag),*(n),"<<arrayPrefix<<"_A,"<<arrayPrefix<<"_X,*(incx));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

