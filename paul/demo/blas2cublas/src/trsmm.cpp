#include "blas2cublas.h"

using namespace std;

void handleTRSMM(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){

	ostringstream cocciStream;
	string matARef = "";
	string matBRef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	string cuTrans = "";
	string cuUplo = "";
	string cblasSide = "";
	string cuDiag = "";

	string sideA = "";
	string cblasUplo = "";
	string cblasTrans = "";
	string cblasDiag = "";

	if(checkBlasCallType){
		sideA = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasUplo = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(3)->unparseToString();
		cblasDiag = fArgs->get_traversalSuccessorByIndex(4)->unparseToString();
	}

	else{
		sideA = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
		cblasDiag = fArgs->get_traversalSuccessorByIndex(3)->unparseToString();
	}


	if(sideA == "CblasLeft")       cblasSide = "\'L\'";
	else if(sideA == "CblasRight") cblasSide = "\'R\'";

	if(    cblasTrans  == "CblasTrans")     cuTrans = "\'T\'";
	else if(cblasTrans == "CblasNoTrans")   cuTrans = "\'N\'";
	else if(cblasTrans == "CblasConjTrans") cuTrans = "\'C\'";

	if(cblasUplo == "CblasUpper") cuUplo = "\'U\'";
	else if(cblasUplo == "CblasLower") cuUplo = "\'L\'";

	if(cblasDiag == "CblasNonUnit") cuDiag = "\'N\'";
	else if(cblasDiag == "CblasUnit") cuDiag = "\'U\'";

	SgNode* matrixAptr = NULL;
	SgNode* matrixBptr = NULL;

	if(checkBlasCallType){
		matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(10);
	}
	else{
		matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(9);
	}

	matARef = matrixAptr->unparseToCompleteString();
	matBRef = matrixBptr->unparseToCompleteString();

	if(fname.find("strmm") != string::npos){
		aType = "float";
		cublasCall = "cublasStrmm";
	}
	else if(fname.find("dtrmm") != string::npos){
		aType = "double";
		cublasCall = "cublasDtrmm";
	}
	else if(fname.find("ctrmm") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtrmm";
	}
	else if(fname.find("ztrmm") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtrmm";
	}
	else if(fname.find("strsm") != string::npos){
		aType = "float";
		cublasCall = "cublasStrsm";
	}
	else if(fname.find("dtrsm") != string::npos){
		aType = "double";
		cublasCall = "cublasDtrsm";
	}
	else if(fname.find("ctrsm") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCtrsm";
	}
	else if(fname.find("ztrsm") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZtrsm";
	}


	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,side,uplo,transa,diag;  \n";
	cocciStream << "expression m,n,lda,alpha,ldb;  \n";
	cocciStream << "@@ \n";
	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,side,uplo,transa,diag,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb);  \n";
	else cocciStream <<   "- "<<blasCall<<"(side,uplo,transa,diag,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb);  \n";


	cocciStream << "+ \n";
	cocciStream << "+  /* Allocate device memory */  \n";


	DeclareDevicePtrB3(cocciStream,aType,arrayPrefix,true,true,false);

	string cA="";
	string cuSide = cblasSide;

	if(cblasSide == "\'L\'" || cblasSide == "")
	{
		cA = "m";
		if(cblasSide == "") {
			cuSide = "side";
			cocciStream << "//Warning:CBLAS_SIDE could not be determined. Default = \'L\' \n"	;
		}
		cocciStream << "+  cublasAlloc(lda*m, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	}
	else if(cblasSide == "\'R\'"){
		cA = "n";
		cocciStream << "+  cublasAlloc(lda*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	}

	if(cblasUplo == "") {
		cocciStream << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cuUplo = "uplo";
	}

	if(cblasTrans == ""){
		cuTrans = "transa";
		cocciStream << "//Warning:CBLAS_TRANS could not be determined. Default = \'N\' \n";
	}

	if(cblasDiag == ""){
		cuDiag = "diag";
		cocciStream << "//Warning:CBLAS_DIAG could not be determined. Default = \'N\' \n";
	}

	cocciStream << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_B);  \n";
	cocciStream << "+       \n";

	cocciStream << "+  /* Copy matrices to device */   \n";
	cocciStream << "+  cublasSetMatrix (lda,"<< cA<<", sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<",lda, (void *) "<<arrayPrefix<<"_A, lda);  \n";
	cocciStream << "+  cublasSetMatrix (m, n, sizeType_"<<arrayPrefix<<", (void *)"<<matBRef<<",m, (void *) "<<arrayPrefix<<"_B,m);  \n";
	cocciStream << "+     \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);
	cocciStream << "+  "<<cublasCall<<"("<<cuUplo<<","<<cuTrans<<","<<cuDiag<<",m,n,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result array back to host */ \n";
	cocciStream << "+  cublasSetMatrix( m, n, sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_B, m, (void *)"<<matBRef<<", m); \n";

	FreeDeviceMemoryB3(cocciStream,arrayPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

