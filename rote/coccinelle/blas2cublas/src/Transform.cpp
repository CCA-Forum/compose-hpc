#include "Transform.h"
#include "Blas2Cublas.h"

using namespace SageInterface;

void handleBlasCalls(ofstream&, string&, SgExprListExp *, string, int *);
void cublasHeaderInsert(ofstream&);

bool fileExists(const std::string& filename) {
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1) {
        return true;
    }
    return false;
}

//Validate BLAS to CUBLAS annotation options.

BlasToCublasTransform::BlasToCublasTransform(KVAnnotationValue *val,
        SgNode *p) {
    root = p;
    // Get prefix - which is the user provided prefix (to avoid name
    // clashes in transformed code) for new variables introduced
    // (that to point to gpu memory) as part of the transformation.
    // Report error and quit if prefix is not provided.
    // make sure it is a key-value annotation!
    val = isKVAnnotationValue(val);
    ROSE_ASSERT(val != NULL);
    cout << "Found annotated node:" << p->class_name() << endl;
    Dynamic *chkPrefix = val->lookup("prefix");

    if (chkPrefix != NULL) {
        arrayPrefix = chkPrefix->string_value();
        cout << "Prefix specified: " << arrayPrefix << endl;
    } else {
        cerr
                << "BLAS To CUBLAS transformation error : variable prefix not specified. "
                << endl;
        exit(1);
    }
}

void BlasToCublasTransform::generate(string inpFile, int *fileCount, int *firstBlas) {
    *firstBlas += 1;
    cout << root->unparseToCompleteString() << endl;

    // Get the node (blas call) associated with the annotation.
    SgExprStatement *es = isSgExprStatement(root);
    SgExpression *fCall = es->get_expression();
    // Ensure it is indeed a function call node.
    SgFunctionCallExp *funCall = isSgFunctionCallExp(fCall);
    // Get name of the blas routine.
    SgFunctionSymbol *funcSym = funCall->getAssociatedFunctionSymbol();
    SgName funcName = funcSym->get_name();
    string &fname = (&funcName)->getString();

    // Get argument list for the annotated blas call.
    SgExprListExp *fArgs = funCall->get_args();
    // Get number of arguments.
    size_t nArgs = fArgs->get_numberOfTraversalSuccessors();

    // Generate name for the coccinelle rules
    // file that needs to be generated.
    // inpFile provides the input source file name
    // that is to be transformed.
    string cocciFile = inpFile + "_blasCalls.cocci";

    // File pointer for the coccinelle rules file.
    ofstream cocciFptr;

    // The coccinelle rules file does not exist or
    // was not already created when handling the
    // first annotated BLAS call, which means
    // this is the first annotated BLAS call that
    // is being processed.
    if (!fileExists(cocciFile) || *fileCount == 0) {
        //So create the file
        cocciFptr.open(cocciFile.c_str());
        // Insert header include (cublas_v2.h) rules
        // just once when the coccinelle rules
        // file is created.
        cublasHeaderInsert(cocciFptr);
        //Reset fileCount.
        *fileCount = -1;
    }

    // Coccinelle rules file exists because it was
    // created when handling the intial annotated
    // BLAS call.
    else {
        cocciFptr.open(cocciFile.c_str(), ios::app);
        cocciFptr << "\n\n\n";
    }

    // Main function that identifies the type of BLAS routine
    // and calls various other functions that generate the
    // appropriate cocccinelle rules.
    handleBlasCalls(cocciFptr, fname, fArgs, arrayPrefix, firstBlas);

    // Close coccinelle rules file.
    if (cocciFptr.is_open())
        cocciFptr.close();
}

void handleBlasCalls(ofstream &cocciFptr, string &fname, SgExprListExp *fArgs,
        string arrayPrefix, int *firstBlas) {

    size_t npos = string::npos;

    // Identify whether the original BLAS call specified
    // the arrays to be stored in row-major format and generate
    // a warning, since CUDA BLAS assumes column-major storage.
    bool isRowMajor = false;

    // To identify whether the C interface to BLAS is used
    // since this interface allows user to specify whether the
    // arrays are treated to be stored in row/column major format.
    bool checkBlasCallType = (fname.find("cblas") != npos);

    if (checkBlasCallType) {
        // (If possible) Get array storage format specified
        string cblasOrder =
                fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        if (cblasOrder == "CblasRowMajor")
            isRowMajor = true;
    }

    /* --------------- BLAS 3 CALLS -----------------*/

    if (fname.find("scgemm") != npos || fname.find("dzgemm") != npos
            || fname.find("scgemv") != npos || fname.find("dzgemv") != npos) {
        cerr
                << "These routines are only provided by the Intel MKL library\n\
			 and are not handled by the CUDA BLAS library."
                << endl;
    }

    // Handle gemm routines.
    else if (fname.find("gemm") != npos)
        handleGEMM(cocciFptr, checkBlasCallType, isRowMajor, fname, arrayPrefix,
                fArgs, firstBlas);

    // Handle symm and hemm routines.
    else if (fname.find("symm") != npos || fname.find("hemm") != npos)
        handleSYHEMM(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle herk and syrk routines.
    else if (fname.find("herk") != npos || fname.find("syrk") != npos)
        handleSYHERK(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle her2k and syr2k routines.
    else if (fname.find("her2k") != npos || fname.find("syr2k") != npos)
        handleSYHER2K(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle trsm and trmm routines.
    else if (fname.find("trsm") != npos || fname.find("trmm") != npos)
        handleTRSMM(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    /* --------------- BLAS 2 CALLS -----------------*/

    // Handle gbmv routines.
    else if (fname.find("gbmv") != npos)
        handleGBMV(cocciFptr, checkBlasCallType, isRowMajor, fname, arrayPrefix,
                fArgs, firstBlas);

    // Handle gemv routines.
    else if (fname.find("gemv") != npos)
        handleGEMV(cocciFptr, checkBlasCallType, isRowMajor, fname, arrayPrefix,
                fArgs, firstBlas);

    // Handle ger, gerc, geru routines.
    else if (fname.find("ger") != npos)
        handleGER(cocciFptr, checkBlasCallType, isRowMajor, fname, arrayPrefix,
                fArgs, firstBlas);

    // Handle hbmv, sbmv routines.
    else if (fname.find("hbmv") != npos || fname.find("sbmv") != npos)
        handleHSBMV(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle hemv, symv routines.
    else if (fname.find("hemv") != npos || fname.find("symv") != npos)
        handleHSEYMV(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle syr2, her2 routines.
    else if (fname.find("her2") != npos || fname.find("syr2") != npos)
        handleHESYR2(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle syr, her routines.
    else if (fname.find("her") != npos || fname.find("syr") != npos)
        handleHESYR(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle hpmv, spmv routines.
    else if (fname.find("hpmv") != npos || fname.find("spmv") != npos)
        handleHSPMV(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle hpr2, spr2 routines.
    else if (fname.find("hpr2") != npos || fname.find("spr2") != npos)
        handleHSPR2(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle hpr, spr routines.
    else if (fname.find("hpr") != npos || fname.find("spr") != npos)
        handleHSPR(cocciFptr, checkBlasCallType, isRowMajor, fname, arrayPrefix,
                fArgs, firstBlas);

    // Handle tbmv, tbsv routines.
    else if (fname.find("tbmv") != npos || fname.find("tbsv") != npos)
        handleTBSMV(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle tpmv, tpsv routines.
    else if (fname.find("tpmv") != npos || fname.find("tpsv") != npos)
        handleTPSMV(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    // Handle trmv, trsv routines.
    else if (fname.find("trmv") != npos || fname.find("trsv") != npos)
        handleTRSMV(cocciFptr, checkBlasCallType, isRowMajor, fname,
                arrayPrefix, fArgs, firstBlas);

    /* --------------- BLAS 1 CALLS -----------------*/

    // Handle asum, amin, amax, nrm2 routines.
    else if (fname.find("asum") != npos || fname.find("nrm2") != npos
            || fname.find("amin") != npos || fname.find("amax") != npos)

            { /*handleSumNrm2Aminmax(cocciFptr,checkBlasCallType,fname,arrayPrefix,fArgs, firstBlas); */
    }

    // Handle axpy routines.
    else if (fname.find("axpy") != npos)
        handleAXPY(cocciFptr, checkBlasCallType, fname, arrayPrefix, fArgs, firstBlas);

    // Handle axpby routines.
    else if (fname.find("axpby") != npos)
        handleAXPBY(cocciFptr, checkBlasCallType, fname, arrayPrefix, fArgs, firstBlas);

    // Handle copy routines.
    else if (fname.find("copy") != npos)
        handleCOPY(cocciFptr, checkBlasCallType, fname, arrayPrefix, fArgs, firstBlas);

    // Handle dotc, dotu, dot routines.
    else if (fname.find("dot") != npos) { /*handleDOT(cocciFptr,checkBlasCallType,fname,arrayPrefix,fArgs, firstBlas); */
    }

    // Handle scal routines.
    else if (fname.find("scal") != npos)
        handleSCAL(cocciFptr, checkBlasCallType, fname, arrayPrefix, fArgs, firstBlas);

    // Handle swap routines.
    else if (fname.find("swap") != npos)
        handleSWAP(cocciFptr, checkBlasCallType, fname, arrayPrefix, fArgs, firstBlas);

    // Handle rotg, rotmg routines - Do nothing since the CUDA BLAS versions are run on CPU.
    else if (fname.find("rotg") != npos || fname.find("rotmg") != npos) {
    }

    // Handle rotm routines.
    else if (fname.find("rotm") != npos) { /*handleROTM(cocciFptr,checkBlasCallType,fname,arrayPrefix,fArgs, firstBlas); */
    }

    // Handle rot routines.
    else if (fname.find("rot") != npos) { /*handleROT(cocciFptr,checkBlasCallType,fname,arrayPrefix,fArgs, firstBlas); */
    }

    // Report error in an unknown BLAS call is annotated and quit the transformation.
    else {
        cerr << "Unknown BLAS call: " << fname << endl;
        exit(1);
    }
}

void cublasHeaderInsert(ofstream &cocciFptr) {
    // Header (cublas_v2.h) include rules.
    cocciFptr << "//Begin Header insertion patch.\n";

    cocciFptr << "@initialize:python@ \n\n";

    cocciFptr << "first = 0 \n";
    cocciFptr << "second = 0 \n\n";

    cocciFptr << "@first_hdr@ \n";
    cocciFptr << "position p; \n";
    cocciFptr << "@@ \n\n";

    cocciFptr << "#include <...>@p \n\n";

    cocciFptr << "@script:python@ \n";
    cocciFptr << "p << first_hdr.p; \n";
    cocciFptr << "@@ \n\n";

    cocciFptr << "if first == 0: \n";
    cocciFptr << "   print \"keeping first hdr %s\" % (p[0].line) \n";
    cocciFptr << "   first = int(p[0].line) \n";
    cocciFptr << "else: \n";
    cocciFptr << "  print \"dropping first hdr\" \n";
    cocciFptr << "  cocci.include_match(False) \n";

    cocciFptr << "@second_hdr@ \n";
    cocciFptr << "position p; \n";
    cocciFptr << "@@ \n\n";

    cocciFptr << "#include \"...\"@p \n\n";

    cocciFptr << "@script:python@ \n";
    cocciFptr << "p << second_hdr.p; \n";
    cocciFptr << "@@ \n\n";

    cocciFptr << "if int(p[0].line) > first and first != 0: \n";
    cocciFptr << "   print \"dropping second hdr\" \n";
    cocciFptr << "   cocci.include_match(False) \n";
    cocciFptr << "else: \n";
    cocciFptr << "  if second == 0: \n";
    cocciFptr
            << "     print \"keeping second hdr %s because of %d\" % (p[0].line,first) \n";
    cocciFptr << "     second = int(p[0].line) \n";
    cocciFptr << "  else: \n";
    cocciFptr << "     print \"dropping second hdr\" \n";
    cocciFptr << "     cocci.include_match(False) \n\n";

    cocciFptr << "@done@ \n";
    cocciFptr << "position second_hdr.p; \n";
    cocciFptr << "@@ \n\n";

    cocciFptr << "+#include \"cublas_v2.h\" \n";
    cocciFptr << "+#include \"cuda_runtime.h\" \n";
    cocciFptr << "#include \"...\"@p \n\n";

    cocciFptr << "@depends on never done@ \n";
    cocciFptr << "@@ \n\n";

    cocciFptr << "+#include \"cublas_v2.h\" \n";
    cocciFptr << "+#include \"cuda_runtime.h\" \n";

    cocciFptr << "#include <...> \n\n";

    cocciFptr << "//End Header insertion patch.\n\n";
}

