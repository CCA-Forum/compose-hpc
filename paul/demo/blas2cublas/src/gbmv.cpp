#include "blas2cublas.h"

using namespace std;

void handleGBMV(ofstream &cocciFptr,bool checkBlasCallType, bool isRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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
		matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
		vecXptr = fArgs->get_traversalSuccessorByIndex(9);
		vecYptr = fArgs->get_traversalSuccessorByIndex(12);
	}

	else {
		cblasTrans = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
		vecXptr = fArgs->get_traversalSuccessorByIndex(8);
		vecYptr = fArgs->get_traversalSuccessorByIndex(11);

	}

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("sgbmv") != string::npos){
		aType = "float";
		cublasCall = "cublasSgbmv";
	}
	else if(fname.find("dgbmv") != string::npos){
		aType = "double";
		cublasCall = "cublasDgbmv";
	}
	else if(fname.find("cgbmv") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCgbmv";
	}
	else if(fname.find("zgbmv") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZgbmv";
	}

	cocciStream << "@disable paren@ \n";
	cocciStream << "identifier order,trans;  \n";
	cocciStream << "expression m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType)
		cocciStream << "- "<<blasCall<<"(order,trans,m, n, kl, ku, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";
	else cocciStream << "- "<<blasCall<<"(trans,m, n, kl, ku, alpha,"<<matARef<<",lda,"<<vecXRef<<",incx,beta,"<<vecYRef<<",incy); \n";

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
		cocciStream << "+  cublasSetVector ( "<<lenXY<<",, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
		cocciStream << "+  if(beta != 0) cublasSetVector ("<<lenXY<<", sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"("<<cbTrans<<",m, n, kl, ku, alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n\n";

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
		cocciStream << "+  cublasSetVector ( "<<lenXY<<",, sizeType_"<<arrayPrefix<<","<<vecXRef<<", *(incx), "<<arrayPrefix<<"_X, *(incx));  \n";
		cocciStream << "+  if(*(beta) != 0) cublasSetVector ("<<lenXY<<", sizeType_"<<arrayPrefix<<","<<vecYRef<<", *(incy), "<<arrayPrefix<<"_Y, *(incy));  \n\n";

		cocciStream << "+  /* CUBLAS call */  \n";
		cocciStream << "+  "<<cublasCall<<"(*(trans),*(m), *(n), *(kl), *(ku), *(alpha),"<<arrayPrefix<<"_A,*(lda),"<<arrayPrefix<<"_X,*(incx),*(beta),"<<arrayPrefix<<"_Y,*(incy));  \n\n";

		cocciStream << "+  /* Copy result vector back to host */  \n";
		cocciStream << "+  cublasSetVector ( "<<lenXY<<", sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, *(incy), "<<vecYRef<<", *(incy));  \n";
	}

	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

