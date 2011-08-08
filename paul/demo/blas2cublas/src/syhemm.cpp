#include "blas2cublas.h"

using namespace std;

void handleSYHEMM(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){

	ostringstream cocciStream;
	string matARef = "";
	string matBRef = "";
	string matCRef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

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

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,side,uplo;  \n";
	cocciStream << "expression m,n,alpha,a,lda,b,ldb,beta,c,ldc;  \n";
	cocciStream << "@@ \n";
	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,side,uplo,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n\n";
	else cocciStream <<   "- "<<blasCall<<"(side,uplo,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n\n";
	cocciStream << "+  /* Allocate device memory */  \n";
	DeclareDevicePtrB3(cocciStream,aType,arrayPrefix,true,true,true);

	string dimA = "";
	string cblasSide = "";
	string cblasUplo = "";

	if(checkBlasCallType){

		if(sideA == "CblasLeft")       cblasSide = "\'L\'";
		else if(sideA == "CblasRight") cblasSide = "\'R\'";

		if(uploA == "CblasUpper") cblasUplo = "\'U\'";
		else if(uploA == "CblasLower") cblasUplo = "\'L\'";

		if(cblasSide == "\'L\'")
		{
			dimA = "m";
			cocciStream << "+  cublasAlloc(m*m, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		}
		else if(cblasSide == "\'R\'"){
			dimA = "n";
			cocciStream << "+  cublasAlloc(n*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		}

		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_B);  \n";
		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n\n";
		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix ("<<dimA<<","<< dimA<<", sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<","<<dimA<<", (void *) "<<arrayPrefix<<"_A,"<< dimA<<");  \n";
		cocciStream << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<matBRef<<", m, (void *) "<<arrayPrefix<<"_B, m);  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";
		RowMajorWarning(cocciStream,isRowMajor);
		cocciStream << "+  "<<cublasCall<<"("<<cblasSide<<","<<cblasUplo<<",m,n,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb,beta,"<<arrayPrefix<<"_C,ldc);  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( m, n, sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_C, m, (void *)"<<matCRef<<", m); \n";

	}

	else {

		dimA = arrayPrefix + "_dimA";
		cocciStream << "+ int "<<dimA<<"; \n";
		cocciStream << "+ if(*(side) == \'L\') "<<dimA<<" = m; \n";
		cocciStream << "+ else "<<dimA<<" = n; \n\n";

		cocciStream << "+  cublasAlloc("<<dimA<<"*"<< dimA<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";

		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_B);  \n";
		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n\n";
		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix ("<<dimA<<","<< dimA<<", sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<","<<dimA<<", (void *) "<<arrayPrefix<<"_A,"<< dimA<<");  \n";
		cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_"<<arrayPrefix<<", (void *)"<<matBRef<<", *(m), (void *) "<<arrayPrefix<<"_B, *(m));  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";

		cocciStream << "+  "<<cublasCall<<"(*(side),*(uplo),*(m),*(n),*(alpha),"<<arrayPrefix<<"_A,*(lda),"<<arrayPrefix<<"_B,*(ldb),*(beta),"<<arrayPrefix<<"_C,*(ldc));  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( *(m), *(n), sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_C, *(m), (void *)"<<matCRef<<", *(m)); \n";

	}

	FreeDeviceMemoryB3(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

