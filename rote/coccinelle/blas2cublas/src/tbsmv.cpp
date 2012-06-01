#include "Blas2Cublas.h"

using namespace std;

void handleTBSMV(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

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
    string vecXRef = "";

    SgNode* matrixAptr = NULL;
    SgNode* vecXptr = NULL;

    if (checkBlasCallType) {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        cblasTrans = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
        cblasDiag = fArgs->get_traversalSuccessorByIndex(3)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
        vecXptr = fArgs->get_traversalSuccessorByIndex(8);
    }

    else {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        cblasDiag = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
        vecXptr = fArgs->get_traversalSuccessorByIndex(7);
    }

    matARef = matrixAptr->unparseToCompleteString();
    vecXRef = vecXptr->unparseToCompleteString();

    if (fname.find("ctbmv") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCtbmv";
    } else if (fname.find("ztbmv") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZtbmv";
    } else if (fname.find("stbmv") != string::npos) {
        aType = "float";
        cublasCall = "cublasStbmv";
    } else if (fname.find("dtbmv") != string::npos) {
        aType = "double";
        cublasCall = "cublasDtbmv";
    } else if (fname.find("ctbsv") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCtbsv";
    } else if (fname.find("ztbsv") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZtbsv";
    } else if (fname.find("stbsv") != string::npos) {
        aType = "float";
        cublasCall = "cublasStbsv";
    } else if (fname.find("dtbsv") != string::npos) {
        aType = "double";
        cublasCall = "cublasDtbsv";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,uplo,trans,diag;  \n";
    cocciStream << "expression n, k, lda, incx;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,uplo, trans,diag, n, k, "
                << matARef << ",lda," << vecXRef << ",incx); \n";
    else
        cocciStream << "- " << blasCall << "(uplo, trans,diag, n, k," << matARef
                << ",lda," << vecXRef << ",incx); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, true, true, false);
    string handle = "CublasHandle";
    string cudaStat = "CudaStat";
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

    string arrName = "";

    if (checkBlasCallType) {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, n*n * sizeType_" << uPrefix << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, n * sizeType_" << uPrefix << ");  \n\n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vector to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( n,n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", n, (void *) " << uPrefix
                << "_A, n);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n\n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";

        if (cblasUplo == "CblasUpper")
            uplo = "CUBLAS_FILL_MODE_UPPER";
        else if (cblasUplo == "CblasLower")
            uplo = "CUBLAS_FILL_MODE_LOWER";
        else {
            uplo = uPrefix + "_uplo";
            cocciStream << "+ char " << uplo << "; \n";
            cocciStream << "+ if(" << cblasUplo << " == CblasUpper) " << uplo
                    << " = CUBLAS_FILL_MODE_UPPER; \n";
            cocciStream << "+ else " << uplo << " = CUBLAS_FILL_MODE_LOWER; \n";

        }

        if (isRowMajor) {
            if (cblasTrans == "CblasTrans")
                cbTrans = "CUBLAS_OP_N";
            else if (cblasTrans == "CblasNoTrans")
                cbTrans = "CUBLAS_OP_T";
            else if (cblasTrans == "CblasConjTrans")
                cbTrans = "CUBLAS_OP_C";
            else {
                cbTrans = uPrefix + "_trans";
                cocciStream << "+ char " << cbTrans << "; \n";
                cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                        << cbTrans << " = CUBLAS_OP_N; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasNoTrans) " << cbTrans << " = CUBLAS_OP_T; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasConjTrans) " << cbTrans
                        << " = CUBLAS_OP_C; \n\n";

            }
        } else {
            if (cblasTrans == "CblasTrans")
                cbTrans = "CUBLAS_OP_T";
            else if (cblasTrans == "CblasNoTrans")
                cbTrans = "CUBLAS_OP_N";
            else if (cblasTrans == "CblasConjTrans")
                cbTrans = "CUBLAS_OP_C";
            else {
                cbTrans = uPrefix + "_trans";
                cocciStream << "+ char " << cbTrans << "; \n";
                cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                        << cbTrans << " = CUBLAS_OP_T; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasNoTrans) " << cbTrans << " = CUBLAS_OP_N; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasConjTrans) " << cbTrans
                        << " = CUBLAS_OP_C; \n\n";

            }
        }

        if (cblasDiag == "CblasNonUnit")
            diag = "CUBLAS_DIAG_NON_UNIT";
        else if (cblasDiag == "CblasUnit")
            diag = "CUBLAS_DIAG_UNIT";
        else {
            diag = uPrefix + "_diag";
            cocciStream << "+ char " << diag << "; \n";
            cocciStream << "+ if(" << cblasDiag << " == CblasUnit) " << diag
                    << " = CUBLAS_DIAG_UNIT; \n";
            cocciStream << "+ else " << diag << " = CUBLAS_DIAG_NON_UNIT; \n";

        }

        cocciStream << "+  " << stat << " = " << cublasCall << "(" << uplo << "," << cbTrans
                << "," << diag << ",n,k," << uPrefix << "_A,lda," << uPrefix
                << "_X,incx);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector ( n, sizeType_" << uPrefix << ","
                << uPrefix << "_X, incx, " << vecXRef << ", incx);  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

    }

    else {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, *(n) * *(n) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vector to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(n),*(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(n), (void *) " << uPrefix
                << "_A, *(n));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n\n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";

        cocciStream << "+  " << stat << " = " << cublasCall
                << "(*(uplo),*(trans),*(diag),*(n),*(k)," << uPrefix
                << "_A, *(lda)," << uPrefix << "_X,*(incx));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector ( *(n), sizeType_" << uPrefix << ","
                << uPrefix << "_X, *(incx), " << vecXRef << ", *(incx));  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, false);
    cocciFptr << cocciStream.str();

}

