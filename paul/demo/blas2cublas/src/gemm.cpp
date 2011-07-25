#include "blas2cublas.h"

using namespace std;

void handleGEMM(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){

	ostringstream cocciStream;
	string matARef = "";
	string matBRef = "";
	string matCRef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	string cbTransA="";
	string cbTransB="";

	string cblasTransA = "";
	string cblasTransB = "";

	if(checkBlasCallType){
		cblasTransA = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		cblasTransB = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
	}

	else {
		cblasTransA = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		cblasTransB = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
	}

	SgNode* matrixAptr = NULL;
	SgNode* matrixBptr = NULL;
	SgNode* matrixCptr = NULL;

	if(    cblasTransA  == "CblasTrans")     cbTransA = "\'T\'";
	else if(cblasTransA == "CblasNoTrans")   cbTransA = "\'N\'";
	else if(cblasTransA == "CblasConjTrans") cbTransA = "\'C\'";

	if(     cblasTransB == "CblasTrans")     cbTransB = "\'T\'";
	else if(cblasTransB == "CblasNoTrans")   cbTransB = "\'N\'";
	else if(cblasTransB == "CblasConjTrans") cbTransB = "\'C\'";

	if(checkBlasCallType){
		matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(9);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(12);
	}
	else {
		matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
		matrixBptr = fArgs->get_traversalSuccessorByIndex(8);
		matrixCptr = fArgs->get_traversalSuccessorByIndex(11);
	}

	matARef = matrixAptr->unparseToCompleteString();
	matBRef = matrixBptr->unparseToCompleteString();
	matCRef = matrixCptr->unparseToCompleteString();

	if(fname.find("sgemm") != string::npos){
		aType = "float";
		cublasCall = "cublasSgemm";
	}
	else if(fname.find("dgemm") != string::npos){
		aType = "double";
		cublasCall = "cublasDgemm";
	}
	else if(fname.find("cgemm") != string::npos){
		//Handling both _cgemm and _cgemm3m calls
		aType = "cuComplex";
		cublasCall = "cublasCgemm";
	}
	else if(fname.find("zgemm") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZgemm";
	}

	cocciStream << "@@ \n";
	cocciStream << "identifier order,transA,transB;  \n";
	cocciStream << "expression rA,cB,cA,alpha,lda,ldb,beta,ldc;  \n";
	cocciStream << "@@ \n";

	if(checkBlasCallType) cocciStream <<   "- "<<blasCall<<"(order,transA,transB,rA,cB,cA,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc); \n";
	else cocciStream <<   "- "<<blasCall<<"(transA,transB,rA,cB,cA,alpha,"<<matARef<<",lda,"<<matBRef<<",ldb,beta,"<<matCRef<<",ldc); \n";

	DeclareDevicePtrB3(cocciStream,aType,arrayPrefix,true,true,true);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(rA*cA, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc(cA*cB, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_B);  \n";
	cocciStream << "+  cublasAlloc(rA*cB, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrices to device */     \n";
	cocciStream << "+  cublasSetMatrix ( rA, cA, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", rA, (void *) "<<arrayPrefix<<"_A, rA);  \n";
	cocciStream << "+  cublasSetMatrix ( cA, cB, sizeType_"<<arrayPrefix<<", (void *)"<<matBRef<<", cA, (void *) "<<arrayPrefix<<"_B, cA);  \n";
	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciStream,warnRowMajor);

	if(cbTransA=="" && cbTransB==""){
		//cocciStream << "+ //WARNING: Transpose Options could not be determined.  \n";
		//cocciStream << "+ //Using non-transposed form of both the input arrays.  \n";
		cocciStream << "+  "<<cublasCall<<"(transA,transB,rA,cB,cA,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb,beta,"<<arrayPrefix<<"_C,ldc);  \n";
	}
	else if(cbTransA==""){
		//cocciStream << "+ //WARNING: Transpose Options for array \'A\' could not be determined.  \n";
		//cocciStream << "+ //Assuming non-transposed form for the array \'A\'.  \n";
		cocciStream << "+  "<<cublasCall<<"(transA,"<<cbTransB<<",rA,cB,cA,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb,beta,"<<arrayPrefix<<"_C,ldc);  \n";
	}
	else if(cbTransB==""){
		//cocciStream << "+ //WARNING: Transpose Options for array \'B\' could not be determined.  \n";
		//cocciStream << "+ //Assuming non-transposed form for the array \'B\'.  \n";
		cocciStream << "+  "<<cublasCall<<"("<<cbTransA<<",transB,rA,cB,cA,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb,beta,"<<arrayPrefix<<"_C,ldc);  \n";
	}
	else
		cocciStream << "+  "<<cublasCall<<"("<<cbTransA<<","<<cbTransB<<",rA,cB,cA,alpha,"<<arrayPrefix<<"_A,lda,"<<arrayPrefix<<"_B,ldb,beta,"<<arrayPrefix<<"_C,ldc);\n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result array back to host */  \n";
	cocciStream << "+  cublasSetMatrix( rA, cB, sizeType_"<<arrayPrefix<<", (void *) "<<arrayPrefix<<"_C, rA, (void *)"<<matCRef<<", rA);  \n";
	FreeDeviceMemoryB3(cocciStream,arrayPrefix,true,true,true);
	cocciFptr << cocciStream.str();

}

