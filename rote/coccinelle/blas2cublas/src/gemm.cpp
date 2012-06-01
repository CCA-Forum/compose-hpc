#include "Blas2Cublas.h"

using namespace std;

void handleGEMM(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

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

    string handle = "CublasHandle";
    string cudaStat = "CudaStat";
    string alpha = "alpha_" + uPrefix;
    string beta = "beta_" + uPrefix;
    string stat = "CudaStatReturn";

    if (*firstBlas == 1) {

        cocciStream << "+ cublasHandle_t " << handle << "; \n";
        cocciStream << "+ cublasStatus_t " << stat << " = cublasCreate(&"
                << handle << "); \n";
        cocciStream << "+ cudaError_t " << cudaStat << "; \n";
        cocciStream << "+  \n";
        cocciStream << "+ if( " << stat << " != CUBLAS_STATUS_SUCCESS ) { \n";
        cocciStream
                << "+        printf ( \"CUBLAS initialization failed \\n\" ); \n";
        cocciStream << "+        return EXIT_FAILURE; \n";
        cocciStream << "+  } \n\n";
        cocciStream << "+  \n";
        cocciStream << "+ // Move and uncomment the following handle destroy call to the end of your cuda code. \n";
        cocciStream << "+ // cublasDestroy(&" << handle << "); \n";
        cocciStream << "+  \n";
    }

    cocciStream << "+  "<<aType<<" "<< alpha << " = alpha; \n";
    cocciStream << "+  "<<aType<<" " << beta << " = beta; \n";
    cocciStream << "+  \n";
    string arrName = "";

    if (checkBlasCallType) {
        // C BLAS interface is used
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, rA*cA*sizeType_"
                << uPrefix << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B, cA*cB*sizeType_"
                << uPrefix << ");  \n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_C, rA*cB*sizeType_"
                << uPrefix << ");  \n\n";
        arrName = uPrefix+"_C";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrices to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( rA, cA, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", rA, (void *) " << uPrefix
                << "_A, rA);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix ( cA, cB, sizeType_" << uPrefix
                << ", (void *)" << matBRef << ", cA, (void *) " << uPrefix
                << "_B, cA);  \n\n";
        cocciStream << "+  \n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";

        string cbTransA = "";
        string cbTransB = "";

        if (cblasTransA == "CblasTrans")
            cbTransA = "CUBLAS_OP_T";
        else if (cblasTransA == "CblasNoTrans")
            cbTransA = "CUBLAS_OP_N";
        else if (cblasTransA == "CblasConjTrans")
            cbTransA = "CUBLAS_OP_C";
        else {
            cbTransA = uPrefix + "_transA";
            cocciStream << "+ char " << cbTransA << "; \n";
            cocciStream << "+ if(" << cblasTransA << " == CblasTrans) "
                    << cbTransA << " = CUBLAS_OP_T; \n";
            cocciStream << "+ else if(" << cblasTransA << " == CblasNoTrans) "
                    << cbTransA << " = CUBLAS_OP_N; \n";
            cocciStream << "+ else if(" << cblasTransA << " == CblasConjTrans) "
                    << cbTransA << " = CUBLAS_OP_C; \n\n";

        }

        if (cblasTransB == "CblasTrans")
            cbTransB = "CUBLAS_OP_T";
        else if (cblasTransB == "CblasNoTrans")
            cbTransB = "CUBLAS_OP_N";
        else if (cblasTransB == "CblasConjTrans")
            cbTransB = "CUBLAS_OP_C";
        else {
            cbTransB = uPrefix + "_transB";
            cocciStream << "+ char " << cbTransB << "; \n";
            cocciStream << "+ if(" << cblasTransB << " == CblasTrans) "
                    << cbTransB << " = CUBLAS_OP_T; \n";
            cocciStream << "+ else if(" << cblasTransB << " == CblasNoTrans) "
                    << cbTransB << " = CUBLAS_OP_N; \n";
            cocciStream << "+ else if(" << cblasTransB << " == CblasConjTrans) "
                    << cbTransB << " = CUBLAS_OP_C; \n\n";
        }

        if (isRowMajor) {
            cocciStream << "+ " << stat << " = " << cublasCall << "(" << handle << ","
                    << cbTransA << "," << cbTransB << ",cB,rA,cA,&"<<alpha<<","
                    << uPrefix << "_B,cB," << uPrefix << "_A,cA,&"<<beta<<","
                    << uPrefix << "_C,cB);\n\n";
            blasSuccessCheck(cocciStream,cublasCall);
        } else {
            cocciStream << "+ " << stat << " = " << cublasCall << "(" << handle << ","
                    << cbTransA << "," << cbTransB << ",rA,cB,cA,&"<<alpha<<","
                    << uPrefix << "_A,lda," << uPrefix << "_B,ldb,&"<<beta<<","
                    << uPrefix << "_C,ldc);\n\n";
            blasSuccessCheck(cocciStream,cublasCall);
        }

        cocciStream << "+  \n";
        cocciStream << "+  /* Copy result array back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( rA, cB, sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, rA, (void *)" << matCRef
                << ", rA);  \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);
        cocciStream << "+  \n";
    }

    else {
        // F77 BLAS interface is used.
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, *(rA) * *(cA) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B, *(cA) * *(cB) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_C, *(rA) * *(cB) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_C";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  \n";
        cocciStream << "+  /* Copy matrices to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(rA), *(cA), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(rA), (void *) " << uPrefix
                << "_A, *(rA));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(cA), *(cB), sizeType_" << uPrefix
                << ", (void *)" << matBRef << ", *(cA), (void *) " << uPrefix
                << "_B, *(cA));  \n\n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  \n";
        cocciStream << "+  /* CUBLAS Call */  \n";
        cocciStream << "+ " << stat << " = " << cublasCall << "(" << handle
                << ",*(transA),*(transB),*(rA),*(cB),*(cA),*(alpha)," << uPrefix
                << "_A,*(lda)," << uPrefix << "_B,*(ldb),*(beta)," << uPrefix
                << "_C,*(ldc));\n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  \n";
        cocciStream << "+  /* Copy result array back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( *(rA), *(cB), sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, *(rA), (void *)" << matCRef
                << ", *(rA));  \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  \n";

    }
    FreeDeviceMemoryB3(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

