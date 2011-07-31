#include "blas2cublas.h"

using namespace std;

void handleSYHERK(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){

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

	else{
		cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
	}

	if(    cblasTrans  == "CblasTrans")     cuTrans = "\'T\'";
	else if(cblasTrans == "CblasNoTrans")   cuTrans = "\'N\'";
	else if(cblasTrans == "CblasConjTrans") cuTrans = "\'C\'";


	if(cblasUplo == "CblasUpper") cuUplo = "\'U\'";
	else if(cblasUplo == "CblasLower") cuUplo = "\'L\'";


	SgNode* matrixAptr = NULL;
	SgNode* matrixCptr = NULL;


	if(checkBlasCallType){

		matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(9);
	}
	else{
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(8);
	}

	matARef = matrixAptr->unparseToCompleteString();
	matCRef = matrixCptr->unparseToCompleteString();

	if(fname.find("cherk") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCherk";
	}
	else if(fname.find("zherk") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZherk";
	}
	else if(fname.find("ssyrk") != string::npos){
		aType = "float";
		cublasCall = "cublasSsyrk";
	}
	else if(fname.find("dsyrk") != string::npos){
		aType = "double";
		cublasCall = "cublasDsyrk";
	}
	else if(fname.find("csyrk") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCsyrk";
	}
	else if(fname.find("zsyrk") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZsyrk";
	}
	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,uplo,trans;  \n";
	cocciStream << "expression n,k,alpha,a,lda,beta,c,ldc;  \n";
	cocciStream << "@@ \n";
	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,uplo,trans,n,k,alpha,"<<matARef<<",lda,beta,"<<matCRef<<",ldc);  \n";
	else cocciStream <<   "- "<<blasCall<<"(uplo,trans,n,k,alpha,"<<matARef<<",lda,beta,"<<matCRef<<",ldc);  \n";
	cocciStream << "+ \n";
	cocciStream << "+  /* Allocate device memory */  \n";
	DeclareDevicePtrB3(cocciStream,aType,arrayPrefix,true,false,true);


	string rA = "";
	string cA = "";
	string dimC = "n";

	if(cblasUplo == "") {
		cocciStream << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cuUplo = "uplo";
	}

	if(cblasTrans == "CblasNoTrans" || cblasTrans == "")
	{
		rA = "n";
		cA = "k";
		if(cblasTrans == ""){
			cuTrans = "trans";
			cocciStream << "//Warning:CBLAS_TRANS could not be determined. Default = \'N\' \n";
		}
	}
	else if(cblasTrans == "CblasTrans" || cblasTrans == "CblasConjTrans"){
		rA = "k";
		cA = "n";
	}

	cocciStream << "+  cublasAlloc(n*k, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n";
	cocciStream << "+       \n";

	cocciStream << "+  /* Copy matrices to device */   \n";
	cocciStream << "+  cublasSetMatrix ("<<rA<<","<< cA<<", sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<","<<rA<<", (void *) "<<arrayPrefix<<"_A,"<< rA<<");  \n";
	cocciStream << "+     \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);
	cocciStream << "+  "<<cublasCall<<"("<<cuUplo<<","<<cuTrans<<",n,k,alpha,"<<arrayPrefix<<"_A,lda,beta,"<<arrayPrefix<<"_C,ldc);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result array back to host */ \n";
	cocciStream << "+  cublasSetMatrix( n, n, sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_C, n, (void *)"<<matCRef<<", n); \n";

	FreeDeviceMemoryB3(cocciStream,arrayPrefix,true,false,true);
	cocciFptr << cocciStream.str();

}

