#include "blas2cublas.h"

using namespace std;

void handleSYHER2K(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){

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
	cocciStream << "expression order,uplo,trans;  \n";
	cocciStream << "expression n,k,alpha,a,lda,b,ldb,beta,c,ldc;  \n";
	cocciStream << "@@ \n";
	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,uplo,trans,n,k,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n";
	else cocciStream <<   "- "<<blasCall<<"(uplo,trans,n,k,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc);  \n\n";
	cocciStream << "+  /* Allocate device memory */  \n";
	DeclareDevicePtrB3(cocciStream,aType,uPrefix,true,true,true);

	string rA = "";
	string cA = "";
	string rB = "";
	string cB = "";
	string dimC = "n";

	if(checkBlasCallType){
		if(    cblasTrans  == "CblasTrans") {
			cuTrans = "\'T\'";
			rA = "k"; rB = "k";
			cA = "n"; cB = "n";	
		}
	
		else if(cblasTrans == "CblasNoTrans") 
		{
			rA = "n"; rB = "n";
			cA = "k"; cB = "k";
			cuTrans = "\'N\'";
		}
		else if(cblasTrans == "CblasConjTrans"){
			cuTrans = "\'C\'";
			rA = "k"; rB = "k";
			cA = "n"; cB = "n";
		}
		else{
			cuTrans = uPrefix + "_trans";
			rA = uPrefix + "_rA";
			cocciStream << "+ int "<<rA<<"; \n";
			cA = uPrefix + "_cA";
			cocciStream << "+ int "<<cA<<"; \n";
			rB = uPrefix + "_rB";
			cocciStream << "+ int "<<rB<<"; \n";
			cB = uPrefix + "_cB";
			cocciStream << "+ int "<<cB<<"; \n";
			cocciStream << "+ char "<<cuTrans<<"; \n";
			cocciStream << "+ if("<<cblasTrans<<" == CblasTrans) "<<cuTrans<<" = \'N\'; \n";
			cocciStream << "+ else if("<<cblasTrans<<" == CblasNoTrans) "<<cuTrans<<" = \'T\'; \n";
			cocciStream << "+ else if("<<cblasTrans<<" == CblasConjTrans) "<<cuTrans<<" = \'C\'; \n\n";
			cocciStream << "+ if("<<cuTrans<<" == CblasNoTrans) { "<<rA<<" = n; "<<cA<<" = k; "<<rB<<" = n; "<<cB<<" = k; } \n";
			cocciStream << "+ else { "<<rA<<" = k; "<<cA<<" = n; "<<rB<<" = k; "<<cB<<" = n; } \n\n";
		}

		if(cblasUplo == "CblasUpper") cuUplo = "\'U\'";
		else if(cblasUplo == "CblasLower") cuUplo = "\'L\'";
		else{
			cuUplo = uPrefix + "_uplo";
			cocciStream << "+ char "<<cuUplo<<"; \n";
			cocciStream << "+ if("<<cblasUplo<<" == CblasUpper) "<<cuUplo<<" = \'U\'; \n";
			cocciStream << "+ else "<<cuUplo<<" = \'L\'; \n";

		}

		cocciStream << "+  cublasAlloc(n*k, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n*k, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_B);  \n";
		cocciStream << "+  cublasAlloc(n*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_C);  \n\n";

		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix ("<<rA<<","<< cA<<", sizeType_"<<uPrefix<<", (void *)"<<matARef<<","<<rA<<", (void *) "<<uPrefix<<"_A,"<< rA<<");  \n";
		cocciStream << "+  cublasSetMatrix ("<<rB<<","<< cB<<", sizeType_"<<uPrefix<<", (void *)"<<matBRef<<","<<rB<<", (void *) "<<uPrefix<<"_B,"<< rB<<");  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";
		RowMajorWarning(cocciStream,isRowMajor);
		cocciStream << "+  "<<cublasCall<<"("<<cuUplo<<","<<cuTrans<<",n,k,alpha,"<<uPrefix<<"_A,lda,"<<uPrefix<<"_B,ldb,beta,"<<uPrefix<<"_C,ldc);  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( n, n, sizeType_"<<uPrefix<<", (void *) "<<uPrefix<<"_C, n, (void *)"<<matCRef<<", n); \n";

	}

	else {

		rA = uPrefix + "_rA";
		rB = uPrefix + "_rB";
		cA = uPrefix + "_cA";
		cB = uPrefix + "_cB";

		cocciStream << "+ int "<<rA<<"; \n";
		cocciStream << "+ int "<<rB<<"; \n";
		cocciStream << "+ int "<<cA<<"; \n";
		cocciStream << "+ int "<<cB<<"; \n";

		cocciStream << "+ if(*(trans) == \'N\') { "<<rA<<" = n; "<<rB<<" = n; "<<cA<<" = k; "<<cB<<" = k; }\n";
		cocciStream << "+ else { "<<rA<<" = k; "<<rB<<" = k; "<<cA<<" = n; "<<cB<<" = n; }\n\n";

		cocciStream << "+  cublasAlloc(*(n) * *(k), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(k), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_B);  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_C);  \n\n";

		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix ("<<rA<<","<< cA<<", sizeType_"<<uPrefix<<", (void *)"<<matARef<<","<<rA<<", (void *) "<<uPrefix<<"_A,"<< rA<<");  \n";
		cocciStream << "+  cublasSetMatrix ("<<rB<<","<< cB<<", sizeType_"<<uPrefix<<", (void *)"<<matBRef<<","<<rB<<", (void *) "<<uPrefix<<"_B,"<< rB<<");  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";

		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(trans),*(n),*(k),*(alpha),"<<uPrefix<<"_A,*(lda),"<<uPrefix<<"_B,*(ldb),*(beta),"<<uPrefix<<"_C,*(ldc));  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( *(n), *(n), sizeType_"<<uPrefix<<", (void *) "<<uPrefix<<"_C, *(n), (void *)"<<matCRef<<", *(n)); \n";
	}

	FreeDeviceMemoryB3(cocciStream,uPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

