#include "Transform.h"
#include "Tascel.h"

using namespace SageInterface;

void MainTascelTransform(ofstream&, string&, int, SgExprListExp *);

bool fileExists(const std::string& filename) {
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1) {
        return true;
    }
    return false;
}

//Validate TASCEL annotation options.

TascelTransform::TascelTransform(KVAnnotationValue *val, SgNode *p) {
    root = p;
    // Get prefix - which is the user provided prefix (to avoid name
    // clashes in transformed code) for new variables introduced
    // (that to point to gpu memory) as part of the transformation.
    // Report error and quit if prefix is not provided.
    // make sure it is a key-value annotation!
    val = isKVAnnotationValue(val);
    ROSE_ASSERT(val != NULL);
    cout << "Found annotated node:" << p->class_name() << endl;
    Dynamic *chkVersion = val->lookup("version");

    if (chkVersion != NULL) {
        version = atoi(chkVersion->string_value().c_str());
        if (version < 0 || version > 3) {
            cerr
                    << "TASCEL transformation error : unrecognized version specified. "
                    << endl;
            exit(1);
        }
    } else {
        cerr << "TASCEL transformation error : version not specified. " << endl;
        exit(1);
    }
}

void TascelTransform::generate(string inpFile, int *fileCount) {

    // Get the node (next_4chunk call) associated with the annotation.
    SgExprStatement *es = isSgExprStatement(root);

    // Build a list of function calls within the subtree rooted at the annotated node.
    Rose_STL_Container<SgNode*> functionCallList = NodeQuery::querySubTree (root,V_SgFunctionCallExp);

    int counter = 0;
    SgFunctionCallExp* functionCallExp = NULL;
    for (Rose_STL_Container<SgNode*>::iterator i = functionCallList.begin(); i != functionCallList.end(); i++)
    {
        // Build a pointer to the current type so that we can call the get_name() member function.
        functionCallExp = isSgFunctionCallExp(*i);
        counter++;
    }

    ROSE_ASSERT(functionCallExp != NULL);
    // As of now there could be only one function(next_4chunk) call that is annotated.
    //ROSE_ASSERT(counter == 1);

    // Get name of the tascel routine.
    SgFunctionSymbol *funcSym = functionCallExp->getAssociatedFunctionSymbol();
    SgName funcName = funcSym->get_name();
    string &fname = (&funcName)->getString();

    // Get argument list for the call.
    SgExprListExp *fArgs = functionCallExp->get_args();
    // Get number of arguments.
    size_t nArgs = fArgs->get_numberOfTraversalSuccessors();

    // Generate name for the coccinelle rules
    // file that needs to be generated.
    // inpFile provides the input source file name
    // that is to be transformed.
    string cocciFile = inpFile + "_tascel.cocci";

    // File pointer for the coccinelle rules file.
    ofstream cocciFptr;

    // The coccinelle rules file does not exist or
    // was not already created when handling the
    // first annotated (to be transformed) TASCEL call
    if (!fileExists(cocciFile) || *fileCount == 0) {
        //So create the file
        cocciFptr.open(cocciFile.c_str());
        // Insert header include rules
        // just once when the coccinelle rules
        // file is created.
        //tascelHeaderInsert(cocciFptr);
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

    // Main function that identifies the TASCEL routine
    // and calls various other functions that generate the
    // appropriate cocccinelle rules.
    MainTascelTransform(cocciFptr, fname, version, fArgs);

    // Close coccinelle rules file.
    if (cocciFptr.is_open())
        cocciFptr.close();
}

void MainTascelTransform(ofstream &cocciFptr, string &fname, int version,
        SgExprListExp *fArgs) {
    handleTascelTransform(cocciFptr, fname, version, fArgs);
}

void handleTascelTransform(ofstream &cocciFptr, string fname, int version,
        SgExprListExp* fArgs) {
    ostringstream cocciStream;
    if (version == 0) {
        string version0 = "next_4chunk";
        if (fname.compare(version0) == 0) {
            cocciStream << "@@\n";
            cocciStream << "identifier lhs;\n";
            cocciStream << "expression arg1,arg2,arg3,arg4,arg5,arg6;\n";
            cocciStream << "type t;\n";
            cocciStream << "@@\n";
            cocciStream << "<...\n";
            cocciStream << "\n";
            cocciStream << "t lhs;\n";
            cocciStream << "+ long taskid; // ADDED from version 0 -- \n";
            cocciStream << "\n\n";
            cocciStream << "...>\n\n";
            cocciStream << "-lhs=" << fname
                    << "(arg1,arg2,arg3,arg4,arg5,arg6);\n";
            cocciStream
                    << "+taskid = gettask(); // TRANSFORMED from version 0 ---\n";
            cocciStream
                    << "+lhs=translate_task(taskid,arg1,arg2,arg3,arg4,arg5,arg6);\n";
            //cocciStream << "\n...>\n";
            cocciFptr << cocciStream.str();
        } else {
            string arg1 =
                    fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
            string arg2 =
                    fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
            string arg3 =
                    fArgs->get_traversalSuccessorByIndex(3)->unparseToString();

            cocciStream << "@@ \n";
            cocciStream << "type ty; \n";
            cocciStream << "typedef bool; \n";
            cocciStream << "identifier twoel; \n";
            cocciStream << "@@ \n";

            cocciStream << "ty twoel(...){ \n";
            cocciStream << "<... \n";

            cocciStream << "while(...){ \n";
            cocciStream << "... \n";
            cocciStream << fname << "(" << fArgs->unparseToCompleteString()
                    << ");   \n";
            cocciStream << "+// ADDED from version 2 -- \n";
            cocciStream << "+if (!is_task_real(" << arg3 << ", *schwmax, "
                    << arg1 << "," << arg2 << ")) \n";
            cocciStream << "+     continue; // skip fake tasks \n";

            cocciStream << "... \n";
            cocciStream << "} \n";

            cocciStream << "...> \n";
            cocciStream << "} \n\n\n";

            cocciFptr << cocciStream.str();
        }
    } else if (version == 1) {
        string chunk1 = "gettask";
        string chunk2 = "translate_task";
        if (fname.compare(chunk1) == 0) {
            cocciStream << "@@ \n";
            cocciStream << "type t1,t2,ty; \n";
            cocciStream << "identifier taskid,dotask,twoel; \n";
            cocciStream << "expression arg1,arg2,arg3,arg4,arg5,arg6; \n";
            cocciStream << "@@ \n";

            cocciStream << "ty twoel(...){ \n";
            cocciStream << "<... \n";
            cocciStream << "t1 dotask;\n";
            cocciStream << "... \n";
            cocciStream << "t2 taskid;\n";
            cocciStream << "+ long imax;\n";
            cocciStream << "+ long tottasks; // ADDED from version 1 -- \n";
            cocciStream << "+ int me; // ADDED from version 1 \n";
            cocciStream << "+ int nproc=0; // ADDED from version 1 \n";
            cocciStream << "+ \n";
            cocciStream << "+ // ADDED from version 1 -- note move to C++ \n";
            cocciStream
                    << "+ //TASCELSEDREP int *map = new int[2 * 2 * nproc]; \n";
            cocciStream << "+ //TASCELSEDREP int *procs = new int[nproc]; \n";
            cocciStream << "+ \n";
            cocciStream << "+ me = GA_Nodeid(); \n";
            cocciStream << "+ \n";
            cocciStream << "+ // ADDED from version 1 --  \n";
            cocciStream << "+ imax = nbfn / ichunk;  \n";
            cocciStream
                    << "+ tottasks = imax * imax * imax * imax; // n^4 total tasks \n";
            cocciStream << "+ \n";
            cocciStream << "... \n";
            cocciStream << "-taskid = " << chunk1 << "();\n";
            cocciStream << "-dotask = " << chunk2
                    << "(taskid,arg1,arg2,arg3,arg4,arg5,arg6);\n";
            cocciStream << "... \n";

            cocciStream << "-while(...){ \n";
            cocciStream << "+for (taskid = 0; taskid < tottasks; taskid++) { \n";
            cocciStream << "+ dotask = TASCELSEDREP + " << chunk2
                    << "(taskid,arg1,arg2,arg3,arg4,arg5,arg6); \n";
            cocciStream << "+ \n";
            cocciStream
                    << "+if (!is_task_local(g_schwarz, arg1,arg2, me, map, procs)) \n";
            cocciStream << "+      continue; \n";
            cocciStream << "... \n";

            cocciStream << "-if (...) \n";
            cocciStream << "accum = 0; \n";

            cocciStream << "... \n";
            cocciStream << "} \n";

            cocciStream << "+ \n";
            cocciStream << "+ // ADDED here version 1 -- \n";
            cocciStream << "+ //TASCELSEDREP delete [] map; \n";
            cocciStream << "+ //TASCELSEDREP delete [] procs; \n";

            cocciStream << "...> \n";
            cocciStream << "} \n\n\n";

            cocciStream << "@@\n";
            cocciStream << "identifier tid;\n";
            cocciStream << "type t;\n";
            cocciStream << "@@\n";
            cocciStream << "<... \n\n";

            cocciStream << "t tid;\n";

            cocciStream << "...> \n\n";

            cocciStream << "-tid = " << fname << "();\n\n\n";

            cocciStream << "@@ \n";
            cocciStream << "type t; \n";
            cocciStream << "typedef bool; \n";
            cocciStream << "@@ \n";
            cocciStream << "t " << fname << "(...) { ... } \n\n";

            cocciStream
                    << "+ bool is_task_local(int g_a, int *lo, int *hi, int me, int *map, int *procs) \n";
            cocciStream << "+ { \n";
            cocciStream
                    << "+  int np = NGA_Locate_region(g_a, lo, hi, map, procs); \n";
            cocciStream << "+  return (procs[0] == me); \n";
            cocciStream << "+ } // is_task_local \n\n\n";

            cocciStream << "+ // ADDED from version 2 \n";
            cocciStream
                    << "+ bool is_task_real(double s_ij[][ichunk], double schwmax, int lo[], int hi[]) \n";
            cocciStream << "+ { \n";
            cocciStream << "+   int i,j; \n";
            cocciStream << "+   for (i = lo[0]; i <= hi[0]; i++) { \n";
            cocciStream << "+     int iloc = i - lo[0]; \n";
            cocciStream << "+     for (j = lo[1]; j <= hi[1]; j++) { \n";
            cocciStream << "+       int jloc = j - lo[1]; \n";
            cocciStream << "+       if (s_ij[iloc][jloc] * schwmax >= tol2e) \n";
            cocciStream << "+ 	return true; \n";
            cocciStream << "+     } \n";
            cocciStream << "+   } \n";
            cocciStream << "+ \n";
            cocciStream << "+   return false; \n";
            cocciStream << "+ } // is_task_real \n";
            cocciFptr << cocciStream.str();
        }

        else if (fname.compare(chunk2) == 0) {
            cocciStream << "@@\n";
            cocciStream << "identifier lhs;\n";
            cocciStream << "expression arg0,arg1,arg2,arg3,arg4,arg5,arg6;\n";
            cocciStream << "type t;\n";
            cocciStream << "@@\n";
            cocciStream << "<...\n";
            cocciStream << "\n";
            cocciStream << "t lhs;\n";
            cocciStream << "...>\n\n";

            cocciStream << "-lhs=" << fname
                    << "(arg0,arg1,arg2,arg3,arg4,arg5,arg6);\n";

            cocciFptr << cocciStream.str();
        }
    }

}
