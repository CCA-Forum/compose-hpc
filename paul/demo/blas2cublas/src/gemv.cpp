#include "blas2cublas.h"

using namespace std;

void handleGEMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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
	cocciStream << "identifier order,trans;  \n";
	cocciStream << "expression m, n, alpha, a, lda, x, incx, beta, y, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,trans,m, n, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciStream << "- "<<blasCall<<"(trans,m, n, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,true,true,true);


	if(checkBlasCallType){

		if(isRowMajor){
			if(    cblasTrans  == "CblasTrans")     cbTrans = "\'N\'";
			else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'T\'";
			else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";
		}
		else{
			if(    cblasTrans  == "CblasTrans")     cbTrans = "\'T\'";
			else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'N\'";
			else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";
		}

		if(cbTrans == "\'N\'") lenXY = "n";
		else if(cbTrans == "\'T\'") lenXY = "m";
		else if(cbTrans == "\'C\'") lenXY = "m";

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", m, (void *) "<<arrayPrefix<<"_A, m);  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
		cocciStream << "+  if(beta != 0) cublasSetVector ( "<<lenXY<<", sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"("<<cbTrans<<",m, n, alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";

	}	

	else{

		lenXY = arrayPrefix + "_lenXY";
		cocciStream << "+ int "<<lenXY<<"; \n";
		cocciStream << "+ if(*(trans) == \'N\') "<<lenXY<<" = n; \n";
		cocciStream << "+ else "<<lenXY<<" = m; \n\n";

		cocciStream << "+  /* Allocate device memory */  \n";
		cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
		cocciStream << "+  cublasAlloc("<<lenXY<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n\n";
		cocciStream << "+  /* Copy matrix, vectors to device */     \n";
		cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", *(m), (void *) "<<arrayPrefix<<"_A, *(m));  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  if(*(beta) != 0) cublasSetVector ( "<<lenXY<<", sizeType_"<<arrayPrefix<<","<<vecYRef<<", *(incy), "<<arrayPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(trans),*(m), *(n), *(alpha),"<<arrayPrefix<<"_A,*(lda),"<<arrayPrefix<<"_X,*(incx),*(beta),"<<arrayPrefix<<"_Y,*(incy));  \n\n";
		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

