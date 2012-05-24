#include "Blas2Cublas.h"

using namespace std;

void handleDOT(ofstream &cocciFptr, bool checkBlasCallType, string fname,
        string uPrefix, SgExprListExp* fArgs) {

    ostringstream cocciStream;

    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
    SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(3);

    string vecXRef = vecXptr->unparseToCompleteString();
    string vecYRef = vecYptr->unparseToCompleteString();

    if (fname.find("cdotc") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCdotc";
    } else if (fname.find("zdotc") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZdotc";
    } else if (fname.find("cdotu") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCdotu";
    } else if (fname.find("zdotu") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZdotu";
    } else if (fname.find("sdot") != string::npos) {
        aType = "float";
        cublasCall = "cublasSdot";
    } else if (fname.find("ddot") != string::npos) {
        aType = "double";
        cublasCall = "cublasDdot";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression n, incx, incy;  \n";
    cocciStream << "@@ \n";

    cocciStream << "<...\n- " << blasCall << "(n, " << vecXRef << ",incx,"
            << vecYRef << ",incy); \n";
    cocciStream << "+ " << aType << " *" << uPrefix << "_result;  \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, false, true, true);

    if (checkBlasCallType) {
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(n, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(n, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_Y);  \n";
        cocciStream << "+  cublasAlloc(1, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_result);  \n\n";

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n";
        cocciStream << "+  cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecYRef << ", incy, " << uPrefix << "_Y, incy);  \n\n";
        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall << "(n, " << uPrefix << "_X,incx,"
                << uPrefix << "_Y,incy);  \n...>\n\n";
    }

    else {
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(*(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(*(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_Y);  \n";
        cocciStream << "+  cublasAlloc(1, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_result);  \n\n";

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n";
        cocciStream << "+  cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecYRef << ", *(incy), " << uPrefix << "_Y, *(incy));  \n\n";
        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall << "(*(n), " << uPrefix
                << "_X,*(incx)," << uPrefix << "_Y,*(incy));  \n...>\n\n";
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, false, true, true);
    cocciStream << "+  cublasFree(" << uPrefix << "_result); \n";
    cocciFptr << cocciStream.str();

}
