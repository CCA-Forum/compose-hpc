#include "blas2cublas.h"

using namespace std;

void handleTPSMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){
	
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
	cocciStream << "expression order,uplo,trans,diag;  \n";
	cocciStream << "expression n, incx;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,uplo, trans,diag, n, "<<matARef<<","<<vecXRef<<",incx); \n";
	else cocciStream << "- "<<blasCall<<"(uplo, trans,diag, n, "<<matARef<<","<<vecXRef<<",incx); \n";

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,true,true,false);

	if(checkBlasCallType){

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(n*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n\n";
		cocciStream << "+  /* Copy matrix, vector to device */     \n";
		cocciStream << "+  cublasSetMatrix ( n,n, sizeType_"<<uPrefix<<", (void *)"<<matARef<<", n, (void *) "<<uPrefix<<"_A, n);  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<vecXRef<<", incx, "<<uPrefix<<"_X, incx);  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";

		if(cblasUplo == "CblasUpper") uplo = "\'U\'";
		else if(cblasUplo == "CblasLower") uplo = "\'L\'";
		else{
			uplo = uPrefix + "_uplo";
			cocciStream << "+ char "<<uplo<<"; \n";
			cocciStream << "+ if("<<cblasUplo<<" == CblasUpper) "<<uplo<<" = \'U\'; \n";
			cocciStream << "+ else "<<uplo<<" = \'L\'; \n";

		}

		if(isRowMajor){
			if(    cblasTrans  == "CblasTrans")     cbTrans = "\'N\'";
			else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'T\'";
			else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";
			else{
				cbTrans = uPrefix + "_trans";
				cocciStream << "+ char "<<cbTrans<<"; \n";
				cocciStream << "+ if("<<cblasTrans<<" == CblasTrans) "<<cbTrans<<" = \'N\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasNoTrans) "<<cbTrans<<" = \'T\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasConjTrans) "<<cbTrans<<" = \'C\'; \n\n";

			}
		}
		else{
			if(    cblasTrans  == "CblasTrans")     cbTrans = "\'T\'";
			else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'N\'";
			else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";
			else{
				cbTrans = uPrefix + "_trans";
				cocciStream << "+ char "<<cbTrans<<"; \n";
				cocciStream << "+ if("<<cblasTrans<<" == CblasTrans) "<<cbTrans<<" = \'T\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasNoTrans) "<<cbTrans<<" = \'N\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasConjTrans) "<<cbTrans<<" = \'C\'; \n\n";

			}
		}

		if(cblasDiag == "CblasNonUnit") diag = "\'N\'";
		else if(cblasDiag == "CblasUnit") diag = "\'U\'";
		else{
			diag = uPrefix + "_diag";
			cocciStream << "+ char "<<diag<<"; \n";
			cocciStream << "+ if("<<cblasDiag<<" == CblasUnit) "<<diag<<" = \'U\'; \n";
			cocciStream << "+ else "<<diag<<" = \'N\'; \n";

		}

		cocciStream << "+  "<<cublasCall<<"("<<uplo<<","<<cbTrans<<","<<diag<<",n,"<<uPrefix<<"_A,"<<uPrefix<<"_X,incx);  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( n, sizeType_"<<uPrefix<<","<<uPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	}

	else{

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc(*(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n\n";
		cocciStream << "+  /* Copy matrix, vector to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(n),*(n), sizeType_"<<uPrefix<<", (void *)"<<matARef<<", *(n), (void *) "<<uPrefix<<"_A, *(n));  \n";
		cocciStream << "+  cublasSetVector ( *(n), sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n\n";
		cocciStream << "+  /* CUBLAS call */  \n";

		cocciStream << "+  "<<cublasCall<<"(*(uplo),*(trans),*(diag),*(n),"<<uPrefix<<"_A,"<<uPrefix<<"_X,*(incx));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector (*(n), sizeType_"<<uPrefix<<","<<uPrefix<<"_X, *(incx), "<<vecXRef<<", *(incx));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,uPrefix,true,true,false);
	cocciFptr << cocciStream.str();

}

