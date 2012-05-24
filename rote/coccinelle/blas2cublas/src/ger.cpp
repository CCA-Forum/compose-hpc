#include "Blas2Cublas.h"

using namespace std;

void handleGER(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs) {

    ostringstream cocciStream;

    string matARef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    string vecXRef = "";
    string vecYRef = "";

    SgNode* matrixAptr = NULL;
    SgNode* vecXptr = NULL;
    SgNode* vecYptr = NULL;

    if (checkBlasCallType) {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
        vecXptr = fArgs->get_traversalSuccessorByIndex(4);
        vecYptr = fArgs->get_traversalSuccessorByIndex(6);
    }

    else {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
        vecXptr = fArgs->get_traversalSuccessorByIndex(3);
        vecYptr = fArgs->get_traversalSuccessorByIndex(5);

    }

    matARef = matrixAptr->unparseToCompleteString();
    vecXRef = vecXptr->unparseToCompleteString();
    vecYRef = vecYptr->unparseToCompleteString();

    if (fname.find("cgerc") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCgerc";
    } else if (fname.find("zgerc") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZgerc";
    } else if (fname.find("cgeru") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCgeru";
    } else if (fname.find("zgeru") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZgeru";
    } else if (fname.find("sger") != string::npos) {
        aType = "float";
        cublasCall = "cublasSger";
    } else if (fname.find("dger") != string::npos) {
        aType = "double";
        cublasCall = "cublasDger";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order;  \n";
    cocciStream << "expression m, n, alpha, incx, incy, lda;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,m, n, alpha," << vecXRef
                << ",incx," << vecYRef << ",incy," << matARef << ",lda); \n";
    else
        cocciStream << "- " << blasCall << "(m, n, alpha," << vecXRef
                << ",incx," << vecYRef << ",incy," << matARef << ",lda); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, true, true, true);

    if (checkBlasCallType) {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(m*n, sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(m, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(n, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_Y);  \n\n";
        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetMatrix ( m, n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", m, (void *) " << uPrefix
                << "_A, m);  \n";
        cocciStream << "+  cublasSetVector ( m, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n";
        cocciStream << "+  cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecYRef << ", incy, " << uPrefix << "_Y, incy);  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        RowMajorWarning(cocciStream, isRowMajor);

        cocciStream << "+  " << cublasCall << "(m, n, alpha," << uPrefix
                << "_X,incx," << uPrefix << "_Y,incy," << uPrefix
                << "_A,lda);  \n\n";
        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+  cublasSetMatrix ( m, n, sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, m, (void *) " << matARef
                << ", m);  \n";
    }

    else {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(*(m), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(*(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_Y);  \n\n";
        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(m), (void *) " << uPrefix
                << "_A, *(m));  \n";
        cocciStream << "+  cublasSetVector ( *(m), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n";
        cocciStream << "+  cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecYRef << ", *(incy), " << uPrefix << "_Y, *(incy));  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall << "(*(m), *(n), *(alpha),"
                << uPrefix << "_X,*(incx)," << uPrefix << "_Y,*(incy),"
                << uPrefix << "_A,*(lda));  \n\n";
        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, *(m), (void *) " << matARef
                << ", *(m));  \n";
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

