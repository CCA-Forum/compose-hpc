#include "blas2cublas.h"

using namespace std;

void handleGBMV(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, string lenX, string lenY, SgExprListExp* fArgs){
	
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

	if(    cblasTrans  == "CblasTrans")     cbTrans = "\'T\'";
	else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'N\'";
	else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";

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

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc("<<lenX<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc("<<lenY<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", m, (void *) "<<arrayPrefix<<"_A, m);  \n";
	cocciStream << "+  cublasSetVector ( "<<lenX<<",, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  if(beta != 0) cublasSetVector ("<<lenY<<", sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);

	if(cbTrans==""){
		//cocciStream << "+ //WARNING: Transpose Options for array \'A\' could not be determined.  \n";
		//cocciStream << "+ //Assuming non-transposed form for the array \'A\'.  \n";
		cocciStream << "+  "<<cublasCall<<"(trans,m, n, kl, ku, 						alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n";
	}

	else
		cocciStream << "+  "<<cublasCall<<"("<<cbTrans<<",m, n, kl, ku, alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_X,incx,beta,"<<arrayPrefix<<"_Y,incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector ( "<<lenY<<", sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

