#include "blas2cublas.h"

using namespace std;

void handleGEMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string uPrefix, SgExprListExp* fArgs){
	
	string lenXY="";

	ostringstream cocciStream;

	string matARef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	string cbTrans="";

	string cblasTrans = "";
	string vecXRef="";
	string vecYRef="";

	SgNode* matrixAptr = NULL;
	SgNode* vecXptr = NULL;
	SgNode* vecYptr = NULL;

	if(checkBlasCallType){
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
		vecXptr = fArgs->get_traversalSuccessorByIndex(7);
		vecYptr = fArgs->get_traversalSuccessorByIndex(10);
	}

	else {
		cblasTrans = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
		vecXptr = fArgs->get_traversalSuccessorByIndex(6);
		vecYptr = fArgs->get_traversalSuccessorByIndex(9);

	}


	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("sgemv") != string::npos){
		aType = "float";
		cublasCall = "cublasSgemv";
	}
	else if(fname.find("dgemv") != string::npos){
		aType = "double";
		cublasCall = "cublasDgemv";
	}
	else if(fname.find("cgemv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCgemv";
	}
	else if(fname.find("zgemv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZgemv";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "expression order,trans;  \n";
	cocciStream << "expression m, n, alpha, a, lda, x, incx, beta, y, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,trans,m, n, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciStream << "- "<<blasCall<<"(trans,m, n, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,uPrefix,true,true,true);


	if(checkBlasCallType){

		if(isRowMajor){
			if(    cblasTrans  == "CblasTrans")    { cbTrans = "\'N\'"; lenXY = "n"; }
			else if(cblasTrans == "CblasNoTrans")  { cbTrans = "\'T\'"; lenXY = "m"; } 
			else if(cblasTrans == "CblasConjTrans") { cbTrans = "\'C\'"; lenXY = "m"; }
			else{
				cbTrans = uPrefix + "_trans";
				lenXY = uPrefix + "_lenXY";
				cocciStream << "+ int "<<lenXY<<"; \n";
				cocciStream << "+ char "<<cbTrans<<"; \n";
				cocciStream << "+ if("<<cblasTrans<<" == CblasTrans) "<<cbTrans<<" = \'N\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasNoTrans) "<<cbTrans<<" = \'T\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasConjTrans) "<<cbTrans<<" = \'C\'; \n\n";
				cocciStream << "+ if("<<cbTrans<<" == \'N\') "<<lenXY<<" = n; \n";
				cocciStream << "+ else "<<lenXY<<" = m; \n\n";

			}
		}
		else{
			if(    cblasTrans  == "CblasTrans")     { cbTrans = "\'T\'"; lenXY = "m"; }
			else if(cblasTrans == "CblasNoTrans")   { cbTrans = "\'N\'"; lenXY = "n"; }
			else if(cblasTrans == "CblasConjTrans") { cbTrans = "\'C\'"; lenXY = "m"; }
			else{
				cbTrans = uPrefix + "_trans";
				lenXY = uPrefix + "_lenXY";
				cocciStream << "+ int "<<lenXY<<"; \n";
				cocciStream << "+ char "<<cbTrans<<"; \n";
				cocciStream << "+ if("<<cblasTrans<<" == CblasTrans) "<<cbTrans<<" = \'T\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasNoTrans) "<<cbTrans<<" = \'N\'; \n";
				cocciStream << "+ else if("<<cblasTrans<<" == CblasConjTrans) "<<cbTrans<<" = \'C\'; \n\n";

				cocciStream << "+ if("<<cbTrans<<" == \'N\') "<<lenXY<<" = n; \n";
				cocciStream << "+ else "<<lenXY<<" = m; \n\n";

			}
		}

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( m, n, sizeType_"<<uPrefix<<", (void *)"<<matARef<<", m, (void *) "<<uPrefix<<"_A, m);  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<uPrefix<<","<<vecXRef<<", incx, "<<uPrefix<<"_X, incx);  \n";
		cocciStream << "+  if(beta != 0) cublasSetVector ( "<<lenXY<<", sizeType_"<<uPrefix<<","<<vecYRef<<", incy, "<<uPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		if(isRowMajor){
			cocciStream << "+  "<<cublasCall<<"("<<cbTrans<<",n, m, alpha,"<<uPrefix<<"_A,lda,"<<uPrefix<<"_X,incx,beta,"<<uPrefix<<"_Y,incy);  \n\n";
		}
		else{
			cocciStream << "+  "<<cublasCall<<"("<<cbTrans<<",m, n, alpha,"<<uPrefix<<"_A,lda,"<<uPrefix<<"_X,incx,beta,"<<uPrefix<<"_Y,incy);  \n\n";
		}
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";

	}	

	else{

		lenXY = uPrefix + "_lenXY";
		cocciStream << "+ int "<<lenXY<<"; \n";
		cocciStream << "+ if(*(trans) == \'N\') "<<lenXY<<" = n; \n";
		cocciStream << "+ else "<<lenXY<<" = m; \n\n";

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<uPrefix<<", (void**)&"<<uPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_"<<uPrefix<<", (void *)"<<matARef<<", *(m), (void *) "<<uPrefix<<"_A, *(m));  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<uPrefix<<","<<vecXRef<<", *(incx), "<<uPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  if(*(beta) != 0) cublasSetVector ( "<<lenXY<<", sizeType_"<<uPrefix<<","<<vecYRef<<", *(incy), "<<uPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(trans),*(m), *(n), *(alpha),"<<uPrefix<<"_A,*(lda),"<<uPrefix<<"_X,*(incx),*(beta),"<<uPrefix<<"_Y,*(incy));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<uPrefix<<","<<uPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,uPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

