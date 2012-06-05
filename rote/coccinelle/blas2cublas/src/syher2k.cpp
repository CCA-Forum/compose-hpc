#include "Blas2Cublas.h"

using namespace std;

void handleSYHER2K(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

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

    if (checkBlasCallType) {

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

    if (checkBlasCallType) {
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

    if (fname.find("cher2k") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCher2k";
    } else if (fname.find("zher2k") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZher2k";
    } else if (fname.find("ssyr2k") != string::npos) {
        aType = "float";
        cublasCall = "cublasSsyr2k";
    } else if (fname.find("dsyr2k") != string::npos) {
        aType = "double";
        cublasCall = "cublasDsyr2k";
    } else if (fname.find("csyr2k") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCsyr2k";
    } else if (fname.find("zsyr2k") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZsyr2k";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,uplo,trans;  \n";
    cocciStream << "expression n,k,alpha,a,lda,b,ldb,beta,c,ldc;  \n";
    cocciStream << "@@ \n";
    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,uplo,trans,n,k,alpha,"
                << matARef << ",lda," << matBRef << ",ldb,beta," << matCRef
                << ",ldc);  \n";
    else
        cocciStream << "- " << blasCall << "(uplo,trans,n,k,alpha," << matARef
                << ",lda," << matBRef << ",ldb,beta," << matCRef
                << ",ldc);  \n\n";
    cocciStream << "+  /* Allocate device memory */  \n";
    DeclareDevicePtrB3(cocciStream, aType, uPrefix, true, true, true);

    string rA = "";
    string cA = "";
    string rB = "";
    string cB = "";
    string dimC = "n";

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

    if (checkBlasCallType) {
        if (cblasTrans == "CblasTrans") {
            cuTrans = "CUBLAS_OP_T";
            rA = "k";
            rB = "k";
            cA = "n";
            cB = "n";
        }

        else if (cblasTrans == "CblasNoTrans") {
            rA = "n";
            rB = "n";
            cA = "k";
            cB = "k";
            cuTrans = "CUBLAS_OP_N";
        } else if (cblasTrans == "CblasConjTrans") {
            cuTrans = "CUBLAS_OP_C";
            rA = "k";
            rB = "k";
            cA = "n";
            cB = "n";
        } else {
            cuTrans = uPrefix + "_trans";
            rA = uPrefix + "_rA";
            cocciStream << "+ int " << rA << "; \n";
            cA = uPrefix + "_cA";
            cocciStream << "+ int " << cA << "; \n";
            rB = uPrefix + "_rB";
            cocciStream << "+ int " << rB << "; \n";
            cB = uPrefix + "_cB";
            cocciStream << "+ int " << cB << "; \n";
            cocciStream << "+ char " << cuTrans << "; \n";
            cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                    << cuTrans << " = CUBLAS_OP_T; \n";
            cocciStream << "+ else if(" << cblasTrans << " == CblasNoTrans) "
                    << cuTrans << " = CUBLAS_OP_N; \n";
            cocciStream << "+ else if(" << cblasTrans << " == CblasConjTrans) "
                    << cuTrans << " = CUBLAS_OP_C; \n\n";
            cocciStream << "+ if(" << cuTrans << " == CblasNoTrans) { " << rA
                    << " = n; " << cA << " = k; " << rB << " = n; " << cB
                    << " = k; } \n";
            cocciStream << "+ else { " << rA << " = k; " << cA << " = n; " << rB
                    << " = k; " << cB << " = n; } \n\n";
        }

        if (cblasUplo == "CblasUpper")
            cuUplo = "CUBLAS_FILL_MODE_UPPER";
        else if (cblasUplo == "CblasLower")
            cuUplo = "CUBLAS_FILL_MODE_LOWER";
        else {
            cuUplo = uPrefix + "_uplo";
            cocciStream << "+ char " << cuUplo << "; \n";
            cocciStream << "+ if(" << cblasUplo << " == CblasUpper) " << cuUplo
                    << " = CUBLAS_FILL_MODE_UPPER; \n";
            cocciStream << "+ else " << cuUplo << " = CUBLAS_FILL_MODE_LOWER; \n";

        }

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, n*k* sizeType_" << uPrefix
                << ");  \n";
        string arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B, n*k* sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_C, n*n* sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_C";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix (" << rA << "," << cA
                << ", sizeType_" << uPrefix << ", (void *)" << matARef << ","
                << rA << ", (void *) " << uPrefix << "_A," << rA << ");  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix (" << rB << "," << cB
                << ", sizeType_" << uPrefix << ", (void *)" << matBRef << ","
                << rB << ", (void *) " << uPrefix << "_B," << rB << ");  \n\n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        RowMajorWarning(cocciStream, isRowMajor);
        cocciStream << "+ " << stat << " = " << cublasCall << "(" << cuUplo << "," << cuTrans
                << ",n,k,&"<<alpha<<"," << uPrefix << "_A,lda," << uPrefix
                << "_B,ldb,&"<<beta<<"," << uPrefix << "_C,ldc);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( n, n, sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, n, (void *)" << matCRef
                << ", n); \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);

    }

    else {

        rA = uPrefix + "_rA";
        rB = uPrefix + "_rB";
        cA = uPrefix + "_cA";
        cB = uPrefix + "_cB";

        cocciStream << "+ int " << rA << "; \n";
        cocciStream << "+ int " << rB << "; \n";
        cocciStream << "+ int " << cA << "; \n";
        cocciStream << "+ int " << cB << "; \n";

        cocciStream << "+ if(*(trans) == \'N\') { " << rA << " = n; " << rB
                << " = n; " << cA << " = k; " << cB << " = k; }\n";
        cocciStream << "+ else { " << rA << " = k; " << rB << " = k; " << cA
                << " = n; " << cB << " = n; }\n\n";

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A,*(n) * *(k) * sizeType_" << uPrefix
                << ");  \n";
        string arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B,*(n) * *(k) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_C, *(n) * *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_C";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix (" << rA << "," << cA
                << ", sizeType_" << uPrefix << ", (void *)" << matARef << ","
                << rA << ", (void *) " << uPrefix << "_A," << rA << ");  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix (" << rB << "," << cB
                << ", sizeType_" << uPrefix << ", (void *)" << matBRef << ","
                << rB << ", (void *) " << uPrefix << "_B," << rB << ");  \n\n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";

        cocciStream << "+ " << stat << " = " << cublasCall
                << "(*(uplo),*(trans),*(n),*(k),*(alpha)," << uPrefix
                << "_A,*(lda)," << uPrefix << "_B,*(ldb),*(beta)," << uPrefix
                << "_C,*(ldc));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( *(n), *(n), sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, *(n), (void *)" << matCRef
                << ", *(n)); \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);
    }

    FreeDeviceMemoryB3(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

