#include "Blas2Cublas.h"

using namespace std;

void handleGEMV(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

    string lenXY = "";

    ostringstream cocciStream;

    string matARef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    string cbTrans = "";

    string cblasTrans = "";
    string vecXRef = "";
    string vecYRef = "";

    SgNode* matrixAptr = NULL;
    SgNode* vecXptr = NULL;
    SgNode* vecYptr = NULL;

    if (checkBlasCallType) {
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

    if (fname.find("sgemv") != string::npos) {
        aType = "float";
        cublasCall = "cublasSgemv";
    } else if (fname.find("dgemv") != string::npos) {
        aType = "double";
        cublasCall = "cublasDgemv";
    } else if (fname.find("cgemv") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCgemv";
    } else if (fname.find("zgemv") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZgemv";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,trans;  \n";
    cocciStream
            << "expression m, n, alpha, a, lda, x, incx, beta, y, incy;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,trans,m, n, alpha,"
                << matARef << ",lda," << vecXRef << ",incx,beta," << vecYRef
                << ",incy); \n";
    else
        cocciStream << "- " << blasCall << "(trans,m, n, alpha," << matARef
                << ",lda," << vecXRef << ",incx,beta," << vecYRef
                << ",incy); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, true, true, true);
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

        if (isRowMajor) {
            if (cblasTrans == "CblasTrans") {
                cbTrans = "CUBLAS_OP_N";
                lenXY = "n";
            } else if (cblasTrans == "CblasNoTrans") {
                cbTrans = "CUBLAS_OP_T";
                lenXY = "m";
            } else if (cblasTrans == "CblasConjTrans") {
                cbTrans = "CUBLAS_OP_C";
                lenXY = "m";
            } else {
                cbTrans = uPrefix + "_trans";
                lenXY = uPrefix + "_lenXY";
                cocciStream << "+ int " << lenXY << "; \n";
                cocciStream << "+ char " << cbTrans << "; \n";
                cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                        << cbTrans << " = CUBLAS_OP_N; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasNoTrans) " << cbTrans << " = CUBLAS_OP_T; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasConjTrans) " << cbTrans
                        << " = CUBLAS_OP_C; \n\n";
                cocciStream << "+ if(" << cbTrans << " == \'N\') " << lenXY
                        << " = n; \n";
                cocciStream << "+ else " << lenXY << " = m; \n\n";

            }
        } else {
            if (cblasTrans == "CblasTrans") {
                cbTrans = "CUBLAS_OP_T";
                lenXY = "m";
            } else if (cblasTrans == "CblasNoTrans") {
                cbTrans = "CUBLAS_OP_N";
                lenXY = "n";
            } else if (cblasTrans == "CblasConjTrans") {
                cbTrans = "CUBLAS_OP_C";
                lenXY = "m";
            } else {
                cbTrans = uPrefix + "_trans";
                lenXY = uPrefix + "_lenXY";
                cocciStream << "+ int " << lenXY << "; \n";
                cocciStream << "+ char " << cbTrans << "; \n";
                cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                        << cbTrans << " = CUBLAS_OP_T; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasNoTrans) " << cbTrans << " = CUBLAS_OP_N; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasConjTrans) " << cbTrans
                        << " = CUBLAS_OP_C; \n\n";

                cocciStream << "+ if(" << cbTrans << " == \'N\') " << lenXY
                        << " = n; \n";
                cocciStream << "+ else " << lenXY << " = m; \n\n";

            }
        }

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, m*n * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, " << lenXY << " * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_Y, " << lenXY << " * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( m, n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", m, (void *) " << uPrefix
                << "_A, m);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( " << lenXY << ", sizeType_"
                << uPrefix << "," << vecXRef << ", incx, " << uPrefix
                << "_X, incx);  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  if(beta != 0) {\n" << stat << " = cublasSetVector ( " << lenXY
                << ", sizeType_" << uPrefix << "," << vecYRef << ", incy, "
                << uPrefix << "_Y, incy);  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);
        cocciStream << "+  }\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        if (isRowMajor) {
            cocciStream << "+  " << stat << " = " << cublasCall << "(" << cbTrans
                    << ",n, m, &"<<alpha<<"," << uPrefix << "_A,lda," << uPrefix
                    << "_X,incx,&"<<beta<<"," << uPrefix << "_Y,incy);  \n\n";
            blasSuccessCheck(cocciStream,cublasCall);
        } else {
            cocciStream << "+  " << stat << " = " << cublasCall << "(" << cbTrans
                    << ",m, n, &"<<alpha<<"," << uPrefix << "_A,lda," << uPrefix
                    << "_X,incx,&"<<beta<<"," << uPrefix << "_Y,incy);  \n\n";
            blasSuccessCheck(cocciStream,cublasCall);
        }
        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector ( " << lenXY << ", sizeType_"
                << uPrefix << "," << uPrefix << "_Y, incy, " << vecYRef
                << ", incy);  \n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);


    }

    else {

        lenXY = uPrefix + "_lenXY";
        cocciStream << "+ int " << lenXY << "; \n";
        cocciStream << "+ if(*(trans) == \'N\') " << lenXY << " = n; \n";
        cocciStream << "+ else " << lenXY << " = m; \n\n";

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, *(m) * *(n) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, " << lenXY << " * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_Y, " << lenXY << " * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(m), (void *) " << uPrefix
                << "_A, *(m));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( " << lenXY << ", sizeType_"
                << uPrefix << "," << vecXRef << ", *(incx), " << uPrefix
                << "_X, *(incx));  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  if(*(beta) != 0) {\n" << stat << " = cublasSetVector ( " << lenXY
                << ", sizeType_" << uPrefix << "," << vecYRef << ", *(incy), "
                << uPrefix << "_Y, *(incy));  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);
        cocciStream << "+  }\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(*(trans),*(m), *(n), *(alpha),"
                << uPrefix << "_A,*(lda)," << uPrefix << "_X,*(incx),*(beta),"
                << uPrefix << "_Y,*(incy));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector ( " << lenXY << ", sizeType_"
                << uPrefix << "," << uPrefix << "_Y, *(incy), " << vecYRef
                << ", *(incy));  \n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);

    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

