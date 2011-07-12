#include "blas2cublas.h"

using namespace std;

void handleSYHEMM(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){

	string matARef = "";
	string matBRef = "";
	string matCRef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";
	string cbTrans="";
	string cblasSide = "";
	string cblasUplo = "";

	string sideA = "";
	string uploA = "";

	if(checkBlasCallType){
		sideA = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		uploA = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
	}

	else{
		sideA = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		uploA = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
	}

	if(sideA == "CblasLeft")       cblasSide = "\'L\'";
	else if(sideA == "CblasRight") cblasSide = "\'R\'";


	if(uploA == "CblasUpper") cblasUplo = "\'U\'";
	else if(uploA == "CblasLower") cblasUplo = "\'L\'";

	SgNode* matrixAptr = NULL;
	SgNode* matrixBptr = NULL;
	SgNode* matrixCptr = NULL;

	if(checkBlasCallType){
		matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(8);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(11);
	}

	else{
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(7);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(10);
	}

	matARef = matrixAptr->unparseToCompleteString();
	matBRef = matrixBptr->unparseToCompleteString();
	matCRef = matrixCptr->unparseToCompleteString();

	if(fname.find("ssymm") != string::npos){
		aType = "float";
		cublasCall = "cublasSsymm";
	}
	else if(fname.find("dsymm") != string::npos){
		aType = "double";
		cublasCall = "cublasDsymm";
	}
	else if(fname.find("csymm") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCsymm";
	}
	else if(fname.find("zsymm") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZsymm";
	}
	else if(fname.find("chemm") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasChemm";
	}
	else if(fname.find("zhemm") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZhemm";
	}

	cocciFptr << "@@ \n";
	cocciFptr << "identifier order,size,uplo;  \n";
	cocciFptr << "expression m,n,alpha,a,lda,b,ldb,beta,c,ldc;  \n";
	cocciFptr << "@@ \n";
	if(checkBlasCallType) cocciFptr <<   "- "<<blasCall<<"(order,size,uplo,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n";
	else cocciFptr <<   "- "<<blasCall<<"(size,uplo,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n";
	cocciFptr << "+ \n";
	cocciFptr << "+  /* Allocate device memory */  \n";
	DeclareDevicePtrB3(cocciFptr,aType,arrayPrefix,true,true,true);

	string dimA = "";
	string cuSide = cblasSide;
	string cuUplo = cblasUplo;

	if(cblasUplo == "") {
		cocciFptr << "//Warning:CBLAS_UPLO could not be determined. Default = \'U\' \n";
		cuUplo = "uplo";
	}

	if(cblasSide == "\'L\'" || cblasSide == "")
	{
		dimA = "m";
		if(cblasSide == "") {
			cuSide = "size";
			cocciFptr << "//Warning:CBLAS_SIDE could not be determined. Default = \'L\' \n"	;
		}
		cocciFptr << "+  cublasAlloc(m*m, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	}
	else if(cblasSide == "\'R\'"){
		dimA = "n";
		cocciFptr << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	}

	cocciFptr << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_B);  \n";
	cocciFptr << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n";
	cocciFptr << "+       \n";
	cocciFptr << "+  /* Copy matrices to device */   \n";
	cocciFptr << "+  cublasSetMatrix ("<<dimA<<","<< dimA<<", sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<","<<dimA<<", (void *) "<<arrayPrefix<<"_A,"<< dimA<<");  \n";
	cocciFptr << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<matBRef<<", m, (void *) "<<arrayPrefix<<"_B, m);  \n";
	cocciFptr << "+     \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciFptr,warnRowMajor);
	cocciFptr << "+  "<<cublasCall<<"("<<cuSide<<","<<cuUplo<<",m,n,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb,beta,"<<arrayPrefix<<"_C,ldc);  \n";
	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result array back to host */ \n";
	cocciFptr << "+  cublasSetMatrix( m, n, sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_C, m, (void *)"<<matCRef<<", m); \n";

	FreeDeviceMemoryB3(cocciFptr,arrayPrefix,true,true,true);

}

