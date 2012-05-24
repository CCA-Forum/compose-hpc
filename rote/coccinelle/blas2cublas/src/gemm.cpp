#include "Blas2Cublas.h"

using namespace std;

void handleGEMM(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs) {

    ostringstream cocciStream;
    string matARef = "";
    string matBRef = "";
    string matCRef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    string cblasTransA = "";
    string cblasTransB = "";

    if (checkBlasCallType) {
        cblasTransA =
                fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        cblasTransB =
                fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
    }

    else {
        cblasTransA =
                fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        cblasTransB =
                fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
    }

    SgNode* matrixAptr = NULL;
    SgNode* matrixBptr = NULL;
    SgNode* matrixCptr = NULL;

    if (checkBlasCallType) {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
        matrixBptr = fArgs->get_traversalSuccessorByIndex(9);
        matrixCptr = fArgs->get_traversalSuccessorByIndex(12);
    } else {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
        matrixBptr = fArgs->get_traversalSuccessorByIndex(8);
        matrixCptr = fArgs->get_traversalSuccessorByIndex(11);
    }

    matARef = matrixAptr->unparseToCompleteString();
    matBRef = matrixBptr->unparseToCompleteString();
    matCRef = matrixCptr->unparseToCompleteString();

    if (fname.find("sgemm") != string::npos) {
        aType = "float";
        cublasCall = "cublasSgemm";
    } else if (fname.find("dgemm") != string::npos) {
        aType = "double";
        cublasCall = "cublasDgemm";
    } else if (fname.find("cgemm") != string::npos) {
        //Handling both _cgemm and _cgemm3m calls
        aType = "cuComplex";
        cublasCall = "cublasCgemm";
    } else if (fname.find("zgemm") != string::npos) {
        //Handling both _zgemm and _zgemm3m calls
        aType = "cuDoubleComplex";
        cublasCall = "cublasZgemm";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,transA,transB;  \n";
    cocciStream << "expression rA,cB,cA,alpha,lda,ldb,beta,ldc;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall
                << "(order,transA,transB,rA,cB,cA,alpha," << matARef << ",lda,"
                << matBRef << ",ldb,beta," << matCRef << ",ldc); \n";
    else
        cocciStream << "- " << blasCall << "(transA,transB,rA,cB,cA,alpha,"
                << matARef << ",lda," << matBRef << ",ldb,beta," << matCRef
                << ",ldc); \n";

    DeclareDevicePtrB3(cocciStream, aType, uPrefix, true, true, true);

    if (checkBlasCallType) {
        // C BLAS interface is used

        string chkAlloc = "chkAlloc_" + uPrefix;
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  int " << chkAlloc << ";\n";
        cocciStream << "+  " << chkAlloc << " = cublasAlloc(rA*cA, sizeType_"
                << uPrefix << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  if(" << chkAlloc << " != CUBLAS_STATUS_SUCCESS) {\n";
        cocciStream
                << "+   	   printf(\"Error allocating memory on device for array "
                << uPrefix << "_A !!\\n\");\n";
        cocciStream << "+          return -1;\n";
        cocciStream << "+  }\n";
        cocciStream << "+  " << chkAlloc << " = cublasAlloc(cA*cB, sizeType_"
                << uPrefix << ", (void**)&" << uPrefix << "_B);  \n";
        cocciStream << "+  if(" << chkAlloc << " != CUBLAS_STATUS_SUCCESS) {\n";
        cocciStream
                << "+   	   printf(\"Error allocating memory on device for array "
                << uPrefix << "_B !!\\n\");\n";
        cocciStream << "+          return -1;\n";
        cocciStream << "+  }\n";
        cocciStream << "+  " << chkAlloc << " = cublasAlloc(rA*cB, sizeType_"
                << uPrefix << ", (void**)&" << uPrefix << "_C);  \n\n";
        cocciStream << "+  if(" << chkAlloc << " != CUBLAS_STATUS_SUCCESS) {\n";
        cocciStream
                << "+   	   printf(\"Error allocating memory on device for array "
                << uPrefix << "_C !!\\n\");\n";
        cocciStream << "+          return -1;\n";
        cocciStream << "+  }\n";
        cocciStream << "+  /* Copy matrices to device */     \n";
        cocciStream << "+  cublasSetMatrix ( rA, cA, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", rA, (void *) " << uPrefix
                << "_A, rA);  \n";
        cocciStream << "+  cublasSetMatrix ( cA, cB, sizeType_" << uPrefix
                << ", (void *)" << matBRef << ", cA, (void *) " << uPrefix
                << "_B, cA);  \n\n";
        cocciStream << "+  /* CUBLAS call */  \n";

        string cbTransA = "";
        string cbTransB = "";

        if (cblasTransA == "CblasTrans")
            cbTransA = "\'T\'";
        else if (cblasTransA == "CblasNoTrans")
            cbTransA = "\'N\'";
        else if (cblasTransA == "CblasConjTrans")
            cbTransA = "\'C\'";
        else {
            cbTransA = uPrefix + "_transA";
            cocciStream << "+ char " << cbTransA << "; \n";
            cocciStream << "+ if(" << cblasTransA << " == CblasTrans) "
                    << cbTransA << " = \'T\'; \n";
            cocciStream << "+ else if(" << cblasTransA << " == CblasNoTrans) "
                    << cbTransA << " = \'N\'; \n";
            cocciStream << "+ else if(" << cblasTransA << " == CblasConjTrans) "
                    << cbTransA << " = \'C\'; \n\n";

        }

        if (cblasTransB == "CblasTrans")
            cbTransB = "\'T\'";
        else if (cblasTransB == "CblasNoTrans")
            cbTransB = "\'N\'";
        else if (cblasTransB == "CblasConjTrans")
            cbTransB = "\'C\'";
        else {
            cbTransB = uPrefix + "_transB";
            cocciStream << "+ char " << cbTransB << "; \n";
            cocciStream << "+ if(" << cblasTransB << " == CblasTrans) "
                    << cbTransB << " = \'T\'; \n";
            cocciStream << "+ else if(" << cblasTransB << " == CblasNoTrans) "
                    << cbTransB << " = \'N\'; \n";
            cocciStream << "+ else if(" << cblasTransB << " == CblasConjTrans) "
                    << cbTransB << " = \'C\'; \n\n";
        }

        if (isRowMajor) {
            cocciStream << "+ " << cublasCall << "(" << cbTransA << ","
                    << cbTransB << ",cB,rA,cA,alpha," << uPrefix << "_B,cB,"
                    << uPrefix << "_A,cA,beta," << uPrefix << "_C,cB);\n\n";
        } else {
            cocciStream << "+ " << cublasCall << "(" << cbTransA << ","
                    << cbTransB << ",rA,cB,cA,alpha," << uPrefix << "_A,lda,"
                    << uPrefix << "_B,ldb,beta," << uPrefix << "_C,ldc);\n\n";
        }

        cocciStream << "+  /* Copy result array back to host */  \n";
        cocciStream << "+  cublasSetMatrix( rA, cB, sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, rA, (void *)" << matCRef
                << ", rA);  \n";
    }

    else {
        // F77 BLAS interface is used.
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(*(rA) * *(cA), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(*(cA) * *(cB), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_B);  \n";
        cocciStream << "+  cublasAlloc(*(rA) * *(cB), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_C);  \n\n";
        cocciStream << "+  /* Copy matrices to device */     \n";
        cocciStream << "+  cublasSetMatrix ( *(rA), *(cA), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(rA), (void *) " << uPrefix
                << "_A, *(rA));  \n";
        cocciStream << "+  cublasSetMatrix ( *(cA), *(cB), sizeType_" << uPrefix
                << ", (void *)" << matBRef << ", *(cA), (void *) " << uPrefix
                << "_B, *(cA));  \n\n";
        cocciStream << "+  /* CUBLAS Call */  \n";
        cocciStream << "+  " << cublasCall
                << "(*(transA),*(transB),*(rA),*(cB),*(cA),*(alpha)," << uPrefix
                << "_A,*(lda)," << uPrefix << "_B,*(ldb),*(beta)," << uPrefix
                << "_C,*(ldc));\n\n";
        cocciStream << "+  /* Copy result array back to host */  \n";
        cocciStream << "+  cublasSetMatrix( *(rA), *(cB), sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, *(rA), (void *)" << matCRef
                << ", *(rA));  \n";

    }
    FreeDeviceMemoryB3(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

