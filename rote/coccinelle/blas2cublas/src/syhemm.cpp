#include "Blas2Cublas.h"

using namespace std;

void handleSYHEMM(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

    ostringstream cocciStream;
    string matARef = "";
    string matBRef = "";
    string matCRef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    string sideA = "";
    string uploA = "";

    if (checkBlasCallType) {
        sideA = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        uploA = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
    }

    else {
        sideA = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        uploA = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
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

    if (fname.find("ssymm") != string::npos) {
        aType = "float";
        cublasCall = "cublasSsymm";
    } else if (fname.find("dsymm") != string::npos) {
        aType = "double";
        cublasCall = "cublasDsymm";
    } else if (fname.find("csymm") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCsymm";
    } else if (fname.find("zsymm") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZsymm";
    } else if (fname.find("chemm") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasChemm";
    } else if (fname.find("zhemm") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZhemm";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,side,uplo;  \n";
    cocciStream << "expression m,n,alpha,a,lda,b,ldb,beta,c,ldc;  \n";
    cocciStream << "@@ \n";
    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,side,uplo,m,n,alpha,"
                << matARef << ",lda," << matBRef << ",ldb,beta," << matCRef
                << ",ldc);  \n\n";
    else
        cocciStream << "- " << blasCall << "(side,uplo,m,n,alpha," << matARef
                << ",lda," << matBRef << ",ldb,beta," << matCRef
                << ",ldc);  \n\n";
    cocciStream << "+  /* Allocate device memory */  \n";
    DeclareDevicePtrB3(cocciStream, aType, uPrefix, true, true, true);

    string dimA = "";
    string cblasSide = "";
    string cblasUplo = "";

    string handle = "CublasHandle";
    string cudaStat = "CudaStat";
    string alpha = "alpha_" + uPrefix;
    string beta = "beta_" + uPrefix;
    string arrName = "";
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

        arrName = uPrefix+"_A";

        if (sideA == "CblasLeft") {
            dimA = "m";
            cblasSide = "CUBLAS_SIDE_LEFT";
            cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, m*m*sizeType_" << uPrefix << ");  \n";
            memAllocCheck(cocciStream, arrName );
        } else if (sideA == "CblasRight") {
            dimA = "n";
            cblasSide = "CUBLAS_SIDE_RIGHT";
            cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, n*n*sizeType_" << uPrefix << ");  \n";
            memAllocCheck(cocciStream, arrName );
        } else {
            cblasSide = uPrefix + "_side";
            dimA = uPrefix + "_dimA";
            cocciStream << "+ int " << dimA << "; \n";
            cocciStream << "+ char " << cblasSide << "; \n";
            cocciStream << "+ if(" << sideA << " == CblasLeft) " << cblasSide
                    << " = CUBLAS_SIDE_LEFT; \n";
            cocciStream << "+ else " << cblasSide << " = CUBLAS_SIDE_RIGHT; \n";
            cocciStream << "+ if(" << cblasSide << " == \'R\') " << dimA
                    << " = n; \n";
            cocciStream << "+ else " << dimA << " = m; \n\n";
            cocciStream << "+ " << cudaStat << " = cudaMalloc( (void**)&" << uPrefix
                    << "_A, " << dimA << " * " << dimA
                    << " * sizeType_" << uPrefix << ");  \n";
            memAllocCheck(cocciStream, arrName );

        }

        if (uploA == "CblasUpper")
            cblasUplo = "CUBLAS_FILL_MODE_UPPER";
        else if (uploA == "CblasLower")
            cblasUplo = "CUBLAS_FILL_MODE_LOWER";

        else {
            cblasUplo = uPrefix + "_uplo";
            cocciStream << "+ char " << cblasUplo << "; \n";
            cocciStream << "+ if(" << uploA << " == CblasUpper) " << cblasUplo
                    << " = CUBLAS_FILL_MODE_UPPER; \n";
            cocciStream << "+ else " << cblasUplo << " = CUBLAS_FILL_MODE_LOWER; \n";

        }

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B,m*n*sizeType_" << uPrefix << ");  \n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_C, m*n*sizeType_" << uPrefix << ");  \n\n";
        arrName = uPrefix+"_C";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix (" << dimA << "," << dimA
                << ", sizeType_" << uPrefix << ", (void *)" << matARef << ","
                << dimA << ", (void *) " << uPrefix << "_A," << dimA
                << ");  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix ( m, n, sizeType_" << uPrefix
                << ", (void *)" << matBRef << ", m, (void *) " << uPrefix
                << "_B, m);  \n\n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        RowMajorWarning(cocciStream, isRowMajor);
        cocciStream << "+ " << stat << " = " << cublasCall << "(" << handle << "," << cblasSide << ","
                << cblasUplo << ",m,n,&"<<alpha<<"," << uPrefix << "_A,lda," << uPrefix
                << "_B,ldb,&"<<beta<<"," << uPrefix << "_C,ldc);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( m, n, sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, m, (void *)" << matCRef
                << ", m); \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);

    }

    else {

        dimA = uPrefix + "_dimA";
        cocciStream << "+ int " << dimA << "; \n";
        cocciStream << "+ if(*(side) == \'L\') " << dimA << " = m; \n";
        cocciStream << "+ else " << dimA << " = n; \n\n";

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, " << dimA << "*" << dimA << "* sizeType_"
                << uPrefix << ");  \n";
        string arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_B, *(m) * *(n) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_B";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " =  cudaMalloc((void**)&" << uPrefix << "_C, *(m) * *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_C";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix (" << dimA << "," << dimA
                << ", sizeType_" << uPrefix << ", (void *)" << matARef << ","
                << dimA << ", (void *) " << uPrefix << "_A," << dimA
                << ");  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << matBRef << ", *(m), (void *) " << uPrefix
                << "_B, *(m));  \n\n";
        arrName = uPrefix+"_B";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";

        cocciStream << "+ " << stat << " = " << cublasCall
                << "(*(side),*(uplo),*(m),*(n),*(alpha)," << uPrefix
                << "_A,*(lda)," << uPrefix << "_B,*(ldb),*(beta)," << uPrefix
                << "_C,*(ldc));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix( *(m), *(n), sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, *(m), (void *)" << matCRef
                << ", *(m)); \n";
        arrName = uPrefix+"_C";
        memCpyCheck(cocciStream, arrName);

    }

    FreeDeviceMemoryB3(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

