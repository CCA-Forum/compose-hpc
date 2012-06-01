#include "Blas2Cublas.h"

using namespace std;

void handleTRSMM(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

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

    if (checkBlasCallType) {
        sideA = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        cblasUplo = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
        cblasTrans = fArgs->get_traversalSuccessorByIndex(3)->unparseToString();
        cblasDiag = fArgs->get_traversalSuccessorByIndex(4)->unparseToString();
    }

    else {
        sideA = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        cblasTrans = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
        cblasDiag = fArgs->get_traversalSuccessorByIndex(3)->unparseToString();
    }

    SgNode* matrixAptr = NULL;
    SgNode* matrixBptr = NULL;

    if (checkBlasCallType) {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
        matrixBptr = fArgs->get_traversalSuccessorByIndex(10);
    } else {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
        matrixBptr = fArgs->get_traversalSuccessorByIndex(9);
    }

    matARef = matrixAptr->unparseToCompleteString();
    matBRef = matrixBptr->unparseToCompleteString();

    if (fname.find("strmm") != string::npos) {
        aType = "float";
        cublasCall = "cublasStrmm";
    } else if (fname.find("dtrmm") != string::npos) {
        aType = "double";
        cublasCall = "cublasDtrmm";
    } else if (fname.find("ctrmm") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCtrmm";
    } else if (fname.find("ztrmm") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZtrmm";
    } else if (fname.find("strsm") != string::npos) {
        aType = "float";
        cublasCall = "cublasStrsm";
    } else if (fname.find("dtrsm") != string::npos) {
        aType = "double";
        cublasCall = "cublasDtrsm";
    } else if (fname.find("ctrsm") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCtrsm";
    } else if (fname.find("ztrsm") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZtrsm";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,side,uplo,transa,diag;  \n";
    cocciStream << "expression m,n,lda,alpha,ldb;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall
                << "(order,side,uplo,transa,diag,m,n,alpha," << matARef
                << ",lda," << matBRef << ",ldb); \n\n";
    else
        cocciStream << "- " << blasCall << "(side,uplo,transa,diag,m,n,alpha,"
                << matARef << ",lda," << matBRef << ",ldb);  \n\n";

    cocciStream << "+  /* Allocate device memory */  \n";

    DeclareDevicePtrB3(cocciStream, aType, uPrefix, true, true, false);

    string cA = "";
    string handle = "CublasHandle";
    string cudaStat = "CudaStat";
    string alpha = "alpha_" + uPrefix;
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
    cocciStream << "+  \n";

    if (checkBlasCallType) {
        string arrName="";
        if (sideA == "CblasLeft") {
            cA = "m";
            cblasSide = "CUBLAS_SIDE_LEFT";
            cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, lda*m* sizeType_" << uPrefix
                    << ");  \n";
            arrName = uPrefix+"_A";
            memAllocCheck(cocciStream, arrName);

        } else if (sideA == "CblasRight") {
            cA = "n";
            cblasSide = "CUBLAS_SIDE_RIGHT";
            cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, lda*n* sizeType_" << uPrefix
                    << ");  \n";
            arrName = uPrefix+"_A";
            memAllocCheck(cocciStream, arrName);

        }

        else {
            cblasSide = uPrefix + "_side";
            cA = uPrefix + "_cA";
            cocciStream << "+ int " << cA << "; \n";
            cocciStream << "+ char " << cblasSide << "; \n";
            cocciStream << "+ if(" << sideA << " == CblasLeft) " << cblasSide
                    << " = CUBLAS_SIDE_LEFT; \n";
            cocciStream << "+ else " << cblasSide << " = CUBLAS_SIDE_RIGHT; \n";
            cocciStream << "+ if(" << cblasSide << " == \'R\') " << cA
                    << " = n; \n";
            cocciStream << "+ else " << cA << " = m; \n\n";
            cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, lda * " << cA << "* sizeType_"
                    << uPrefix << ");  \n";

        }

        if (cblasTrans == "CblasTrans")
            cuTrans = "CUBLAS_OP_T";
        else if (cblasTrans == "CblasNoTrans")
            cuTrans = "CUBLAS_OP_N";
        else if (cblasTrans == "CblasConjTrans")
            cuTrans = "CUBLAS_OP_C";
        else {
            cuTrans = uPrefix + "_trans";
            cocciStream << "+ char " << cuTrans << "; \n";
            cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                    << cuTrans << " = CUBLAS_OP_T; \n";
            cocciStream << "+ else if(" << cblasTrans << " == CblasNoTrans) "
                    << cuTrans << " = CUBLAS_OP_N; \n";
            cocciStream << "+ else if(" << cblasTrans << " == CblasConjTrans) "
                    << cuTrans << " = CUBLAS_OP_C; \n\n";

        }

        if (isRowMajor) {
            RowMajorWarning(cocciStream, true);
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

        if (cblasDiag == "CblasNonUnit")
            cuDiag = "CUBLAS_DIAG_NON_UNIT";
        else if (cblasDiag == "CblasUnit")
            cuDiag = "CUBLAS_DIAG_UNIT";
        else {
            cuDiag = uPrefix + "_diag";
            cocciStream << "+ char " << cuDiag << "; \n";
            cocciStream << "+ if(" << cblasDiag << " == CblasUnit) " << cuDiag
                    << " = CUBLAS_DIAG_UNIT; \n";
            cocciStream << "+ else " << cuDiag << " = CUBLAS_DIAG_NON_UNIT; \n";

        }

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B, m*n* sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);


        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix (lda," << cA << ", sizeType_"
                << uPrefix << ", (void *)" << matARef << ",lda, (void *) "
                << uPrefix << "_A, lda);  \n\n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix (m, n, sizeType_" << uPrefix
                << ", (void *)" << matBRef << ",m, (void *) " << uPrefix
                << "_B,m);  \n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(" << cblasSide << "," << cuUplo
                << "," << cuTrans << "," << cuDiag << ",m,n,&"<<alpha<<"," << uPrefix
                << "_A,lda," << uPrefix << "_B,ldb);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( m, n, sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_B, m, (void *)" << matBRef
                << ", m); \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);

    }

    else {

        cA = uPrefix + "_cA";
        cocciStream << "+ int " << cA << "; \n";
        cocciStream << "+ if(*(side) == \'L\') " << cA << " = m; \n";
        cocciStream << "+ else " << cA << " = n; \n\n";

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, lda*" << cA << "* sizeType_" << uPrefix
                << ");  \n";
        string arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);


        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B, *(m) * *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);


        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix (*(lda)," << cA << ", sizeType_"
                << uPrefix << ", (void *)" << matARef << ",*(lda), (void *) "
                << uPrefix << "_A, *(lda));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix (*(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << matBRef << ",*(m), (void *) " << uPrefix
                << "_B,*(m));  \n\n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall
                << "(*(side),*(uplo),*(transa),*(diag),*(m),*(n),*(alpha),"
                << uPrefix << "_A,*(lda)," << uPrefix << "_B,*(ldb));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( *(m), *(n), sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_B, *(m), (void *)" << matBRef
                << ", *(m)); \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);

    }

    FreeDeviceMemoryB3(cocciStream, uPrefix, true, true, false);
    cocciFptr << cocciStream.str();

}

