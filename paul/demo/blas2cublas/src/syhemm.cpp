#include "blas2cublas.h"

using namespace std;

void handleSYHEMM(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){

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
	cocciStream << "expression order,side,uplo;  \n";
	cocciStream << "expression m,n,alpha,a,lda,b,ldb,beta,c,ldc;  \n";
	cocciStream << "@@ \n";
	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,side,uplo,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n\n";
	else cocciStream <<   "- "<<blasCall<<"(side,uplo,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n\n";
	cocciStream << "+  /* Allocate device memory */  \n";
	DeclareDevicePtrB3(cocciStream,aType,uPrefix,true,true,true);

	string dimA = "";
	string cblasSide = "";
	string cblasUplo = "";

	if(checkBlasCallType){

		if(sideA == "CblasLeft")       
		{
			dimA = "m"; cblasSide = "\'L\'";
			cocciStream << "+  cublasAlloc(m*m, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		}
		else if(sideA == "CblasRight") {
			dimA = "n"; cblasSide = "\'R\'";
			cocciStream << "+  cublasAlloc(n*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		}
		else{
			cblasSide = uPrefix + "_side";
			dimA = uPrefix + "_dimA";
			cocciStream << "+ int "<<dimA<<"; \n";
			cocciStream << "+ char "<<cblasSide<<"; \n";
			cocciStream << "+ if("<<sideA<<" == CblasLeft) "<<cblasSide<<" = \'L\'; \n";
			cocciStream << "+ else "<<cblasSide<<" = \'R\'; \n";
			cocciStream << "+ if("<<cblasSide<<" == \'R\') "<<dimA<<" = n; \n";
			cocciStream << "+ else "<<dimA<<" = m; \n\n";
			cocciStream << "+ cublasAlloc("<<dimA<<" * "<<dimA<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";

		}

		if(uploA == "CblasUpper") cblasUplo = "\'U\'";
		else if(uploA == "CblasLower") cblasUplo = "\'L\'";

		else{
			cblasUplo = uPrefix + "_uplo";
			cocciStream << "+ char "<<cblasUplo<<"; \n";
			cocciStream << "+ if("<<uploA<<" == CblasUpper) "<<cblasUplo<<" = \'U\'; \n";
			cocciStream << "+ else "<<cblasUplo<<" = \'L\'; \n";

		}

		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_B);  \n";
		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_C);  \n\n";
		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix ("<<dimA<<","<< dimA<<", sizeType_"<<uPrefix<<", (void *)"<<matARef<<","<<dimA<<", (void *) "<<uPrefix<<"_A,"<< dimA<<");  \n";
		cocciStream << "+  cublasSetMatrix ( m, n, sizeType_"<<uPrefix<<", (void *)"<<matBRef<<", m, (void *) "<<uPrefix<<"_B, m);  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";
		RowMajorWarning(cocciStream,isRowMajor);
		cocciStream << "+  "<<cublasCall<<"("<<cblasSide<<","<<cblasUplo<<",m,n,alpha,"<<uPrefix<<"_A,lda,"<<uPrefix<<"_B,ldb,beta,"<<uPrefix<<"_C,ldc);  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( m, n, sizeType_"<<uPrefix<<", (void *) "<<uPrefix<<"_C, m, (void *)"<<matCRef<<", m); \n";

	}

	else {

		dimA = uPrefix + "_dimA";
		cocciStream << "+ int "<<dimA<<"; \n";
		cocciStream << "+ if(*(side) == \'L\') "<<dimA<<" = m; \n";
		cocciStream << "+ else "<<dimA<<" = n; \n\n";

		cocciStream << "+  cublasAlloc("<<dimA<<"*"<< dimA<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";

		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_B);  \n";
		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_C);  \n\n";
		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix ("<<dimA<<","<< dimA<<", sizeType_"<<uPrefix<<", (void *)"<<matARef<<","<<dimA<<", (void *) "<<uPrefix<<"_A,"<< dimA<<");  \n";
		cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_"<<uPrefix<<", (void *)"<<matBRef<<", *(m), (void *) "<<uPrefix<<"_B, *(m));  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";

		cocciStream << "+  "<<cublasCall<<"(*(side),*(uplo),*(m),*(n),*(alpha),"<<uPrefix<<"_A,*(lda),"<<uPrefix<<"_B,*(ldb),*(beta),"<<uPrefix<<"_C,*(ldc));  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( *(m), *(n), sizeType_"<<uPrefix<<", (void *) "<<uPrefix<<"_C, *(m), (void *)"<<matCRef<<", *(m)); \n";

	}

	FreeDeviceMemoryB3(cocciStream,uPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

