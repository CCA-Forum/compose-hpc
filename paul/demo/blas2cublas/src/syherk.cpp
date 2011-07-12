#include "blas2cublas.h"

using namespace std;

void handleSYHERK(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){

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
	cocciFptr << "@@ \n";
	cocciFptr << "identifier order,uplo,trans;  \n";
	cocciFptr << "expression n,k,alpha,a,lda,beta,c,ldc;  \n";
	cocciFptr << "@@ \n";
	if(checkBlasCallType) cocciFptr <<   "- "<<blasCall<<"(order,uplo,trans,n,k,alpha,"<<matARef<<",lda,beta,"<<matCRef<<",ldc);  \n";
	else cocciFptr <<   "- "<<blasCall<<"(uplo,trans,n,k,alpha,"<<matARef<<",lda,beta,"<<matCRef<<",ldc);  \n";
	cocciFptr << "+ \n";
	cocciFptr << "+  /* Allocate device memory */  \n";
	DeclareDevicePtrB3(cocciFptr,aType,arrayPrefix,true,false,true);


	string rA = "";
	string cA = "";
	string dimC = "n";

	if(cblasUplo == "") {
		cocciFptr << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cuUplo = "uplo";
	}

	if(cblasTrans == "CblasNoTrans" || cblasTrans == "")
	{
		rA = "n";
		cA = "k";
		if(cblasTrans == ""){
			cuTrans = "trans";
			cocciFptr << "//Warning:CBLAS_TRANS could not be determined. Default = \'N\' \n";
		}
	}
	else if(cblasTrans == "CblasTrans" || cblasTrans == "CblasConjTrans"){
		rA = "k";
		cA = "n";
	}

	cocciFptr << "+  cublasAlloc(n*k, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciFptr << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n";
	cocciFptr << "+       \n";

	cocciFptr << "+  /* Copy matrices to device */   \n";
	cocciFptr << "+  cublasSetMatrix ("<<rA<<","<< cA<<", sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<","<<rA<<", (void *) "<<arrayPrefix<<"_A,"<< rA<<");  \n";
	cocciFptr << "+     \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciFptr,warnRowMajor);
	cocciFptr << "+  "<<cublasCall<<"("<<cuUplo<<","<<cuTrans<<",n,k,alpha,"<<arrayPrefix<<"_A,lda,beta,"<<arrayPrefix<<"_C,ldc);  \n";
	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result array back to host */ \n";
	cocciFptr << "+  cublasSetMatrix( n, n, sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_C, n, (void *)"<<matCRef<<", n); \n";

	FreeDeviceMemoryB3(cocciFptr,arrayPrefix,true,false,true);

}

