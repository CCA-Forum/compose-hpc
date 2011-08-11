#include "blas2cublas.h"

using namespace std;

void handleTRSMM(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){

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
	cocciStream << "expression order,side,uplo,transa,diag;  \n";
	cocciStream << "expression m,n,lda,alpha,ldb;  \n";
	cocciStream << "@@ \n";
	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,side,uplo,transa,diag,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb); \n\n";
	else cocciStream <<   "- "<<blasCall<<"(side,uplo,transa,diag,m,n,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb);  \n\n";

	cocciStream << "+  /* Allocate device memory */  \n";

	DeclareDevicePtrB3(cocciStream,aType,uPrefix,true,true,false);

	string cA="";

	if(checkBlasCallType){

		if(sideA == "CblasLeft")       
		{
			cA = "m"; cblasSide = "\'L\'";
			cocciStream << "+  cublasAlloc(lda*m, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		}
		else if(sideA == "CblasRight"){
			cA = "n"; cblasSide = "\'R\'";
			cocciStream << "+  cublasAlloc(lda*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		} 

		else{
			cblasSide = uPrefix + "_side";
			cA = uPrefix + "_cA";
			cocciStream << "+ int "<<cA<<"; \n";
			cocciStream << "+ char "<<cblasSide<<"; \n";
			cocciStream << "+ if("<<sideA<<" == CblasLeft) "<<cblasSide<<" = \'L\'; \n";
			cocciStream << "+ else "<<cblasSide<<" = \'R\'; \n";
			cocciStream << "+ if("<<cblasSide<<" == \'R\') "<<cA<<" = n; \n";
			cocciStream << "+ else "<<cA<<" = m; \n\n";
			cocciStream << "+  cublasAlloc(lda * "<<cA<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";

		}

		if(isRowMajor){
			if(    cblasTrans  == "CblasTrans")     cuTrans = "\'N\'";
			else if(cblasTrans == "CblasNoTrans")   cuTrans = "\'T\'";
			else if(cblasTrans == "CblasConjTrans") cuTrans = "\'C\'";
			else{
				cuTrans = uPrefix + "_trans";
				cocciStream << "+ char "<<cuTrans<<"; \n";
				cocciStream << "+ if("<<cblasTrans<<" == CblasTrans) "<<cuTrans<<" = \'N\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasNoTrans) "<<cuTrans<<" = \'T\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasConjTrans) "<<cuTrans<<" = \'C\'; \n\n";

			}
		}
		else{
			if(    cblasTrans  == "CblasTrans")     cuTrans = "\'T\'";
			else if(cblasTrans == "CblasNoTrans")   cuTrans = "\'N\'";
			else if(cblasTrans == "CblasConjTrans") cuTrans = "\'C\'";
			else{
				cuTrans = uPrefix + "_trans";
				cocciStream << "+ char "<<cuTrans<<"; \n";
				cocciStream << "+ if("<<cblasTrans<<" == CblasTrans) "<<cuTrans<<" = \'T\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasNoTrans) "<<cuTrans<<" = \'N\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasConjTrans) "<<cuTrans<<" = \'C\'; \n\n";

			}
		}

		if(cblasUplo == "CblasUpper") cuUplo = "\'U\'";
		else if(cblasUplo == "CblasLower") cuUplo = "\'L\'";
		else{
			cuUplo = uPrefix + "_uplo";
			cocciStream << "+ char "<<cuUplo<<"; \n";
			cocciStream << "+ if("<<cblasUplo<<" == CblasUpper) "<<cuUplo<<" = \'U\'; \n";
			cocciStream << "+ else "<<cuUplo<<" = \'L\'; \n";

		}

		if(cblasDiag == "CblasNonUnit") cuDiag = "\'N\'";
		else if(cblasDiag == "CblasUnit") cuDiag = "\'U\'";
		else{
			cuDiag = uPrefix + "_diag";
			cocciStream << "+ char "<<cuDiag<<"; \n";
			cocciStream << "+ if("<<cblasDiag<<" == CblasUnit) "<<cuDiag<<" = \'U\'; \n";
			cocciStream << "+ else "<<cuDiag<<" = \'N\'; \n";

		}

		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_B);  \n\n";

		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix (lda,"<< cA<<", sizeType_"<<uPrefix<<", (void *)"<<matARef<<",lda, (void *) "<<uPrefix<<"_A, lda);  \n\n";
		cocciStream << "+  cublasSetMatrix (m, n, sizeType_"<<uPrefix<<", (void *)"<<matBRef<<",m, (void *) "<<uPrefix<<"_B,m);  \n";
		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"("<<cblasSide<<","<<cuUplo<<","<<cuTrans<<","<<cuDiag<<",m,n,alpha,"<<uPrefix<<"_A,lda,"<<uPrefix<<"_B,ldb);  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( m, n, sizeType_"<<uPrefix<<", (void *) "<<uPrefix<<"_B, m, (void *)"<<matBRef<<", m); \n";

	}

	else{

		cA = uPrefix + "_cA";
		cocciStream << "+ int "<<cA<<"; \n";
		cocciStream << "+ if(*(side) == \'L\') "<<cA<<" = m; \n";
		cocciStream << "+ else "<<cA<<" = n; \n\n";

		cocciStream << "+  cublasAlloc(lda*"<<cA<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";

		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_B);  \n\n";

		cocciStream << "+  /* Copy matrices to device */   \n";
		cocciStream << "+  cublasSetMatrix (*(lda),"<< cA<<", sizeType_"<<uPrefix<<", (void *)"<<matARef<<",*(lda), (void *) "<<uPrefix<<"_A, *(lda));  \n";
		cocciStream << "+  cublasSetMatrix (*(m), *(n), sizeType_"<<uPrefix<<", (void *)"<<matBRef<<",*(m), (void *) "<<uPrefix<<"_B,*(m));  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(side),*(uplo),*(transa),*(diag),*(m),*(n),*(alpha),"<<uPrefix<<"_A,*(lda),"<<uPrefix<<"_B,*(ldb));  \n\n";
		cocciStream << "+  /* Copy result array back to host */ \n";
		cocciStream << "+  cublasSetMatrix( *(m), *(n), sizeType_"<<uPrefix<<", (void *) "<<uPrefix<<"_B, *(m), (void *)"<<matBRef<<", *(m)); \n";

	}

	FreeDeviceMemoryB3(cocciStream,uPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

