#include "blas2cublas.h"

using namespace std;

void handleSYHER2K(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){

	ostringstream cocciStream;
	string matARef = "";
	string matBRef = "";
	string matCRef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	string cuTrans = "";
	string cuUplo = "";
	string cblasUplo = "";
	string cblasTrans = "";

	if(checkBlasCallType){

		cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
	}

	else {
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
	}

	if(    cblasTrans  == "CblasTrans")     cuTrans = "\'T\'";
	else if(cblasTrans == "CblasNoTrans")   cuTrans = "\'N\'";
	else if(cblasTrans == "CblasConjTrans") cuTrans = "\'C\'";


	if(cblasUplo == "CblasUpper") cuUplo = "\'U\'";
	else if(cblasUplo == "CblasLower") cuUplo = "\'L\'";

	SgNode* matrixAptr = NULL;
	SgNode* matrixBptr = NULL;
	SgNode* matrixCptr = NULL;

	if(checkBlasCallType){
		matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(8);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(11);
	}

	else {
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(7);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(10);
	}

	matARef = matrixAptr->unparseToCompleteString();
	matBRef = matrixBptr->unparseToCompleteString();
	matCRef = matrixCptr->unparseToCompleteString();

	if(fname.find("cher2k") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCher2k";
	}
	else if(fname.find("zher2k") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZher2k";
	}
	else if(fname.find("ssyr2k") != string::npos){
		aType = "float";
		cublasCall = "cublasSsyr2k";
	}
	else if(fname.find("dsyr2k") != string::npos){
		aType = "double";
		cublasCall = "cublasDsyr2k";
	}
	else if(fname.find("csyr2k") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCsyr2k";
	}
	else if(fname.find("zsyr2k") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZsyr2k";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo,trans;  \n";
	cocciStream << "expression n,k,alpha,a,lda,b,ldb,beta,c,ldc;  \n";
	cocciStream << "@@ \n";
	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,uplo,trans,n,k,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n";
	else cocciStream <<   "- "<<blasCall<<"(uplo,trans,n,k,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n";
	cocciStream << "+ \n";
	cocciStream << "+  /* Allocate device memory */  \n";


	DeclareDevicePtrB3(cocciStream,aType,arrayPrefix,true,true,true);

	string rA = "";
	string cA = "";
	string rB = "";
	string cB = "";
	string dimC = "n";

	if(cblasUplo == "") {
		cocciStream << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cuUplo = "uplo";
	}

	if(cblasTrans == "CblasNoTrans" || cblasTrans == "")
	{
		rA = "n"; rB = "n";
		cA = "k"; cB = "k";
		if(cblasTrans == ""){
			cuTrans = "trans";
			cocciStream << "//Warning:CBLAS_TRANS could not be determined. Default = \'N\' \n";
		}
	}
	else if(cblasTrans == "CblasTrans" || cblasTrans == "CblasConjTrans"){
		rA = "k"; rB = "k";
		cA = "n"; cB = "n";
	}

	cocciStream << "+  cublasAlloc(n*k, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc(n*k, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_B);  \n";
	cocciStream << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n";
	cocciStream << "+       \n";

	cocciStream << "+  /* Copy matrices to device */   \n";
	cocciStream << "+  cublasSetMatrix ("<<rA<<","<< cA<<", sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<","<<rA<<", (void *) "<<arrayPrefix<<"_A,"<< rA<<");  \n";
	cocciStream << "+  cublasSetMatrix ("<<rB<<","<< cB<<", sizeType_"<<arrayPrefix<<", (void *)"<<matBRef<<","<<rB<<", (void *) "<<arrayPrefix<<"_B,"<< rB<<");  \n";
	cocciStream << "+     \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);
	cocciStream << "+  "<<cublasCall<<"("<<cuUplo<<","<<cuTrans<<",n,k,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb,beta,"<<arrayPrefix<<"_C,ldc);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result array back to host */ \n";
	cocciStream << "+  cublasSetMatrix( n, n, sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_C, n, (void *)"<<matCRef<<", n); \n";

	FreeDeviceMemoryB3(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

