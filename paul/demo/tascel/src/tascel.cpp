/*
 * tascel.cpp
 *
 *  Created on: Oct 8, 2011
 *      Author: Ajay Panyala
 */

#include "tascel.h"

using namespace std;

void handleTascelTransform(ofstream &cocciFptr, string fname, int version, SgExprListExp* fArgs){
	ostringstream cocciStream;
	if(version == 0){
		string version0 = "next_4chunk";
		if(fname.compare(version0)==0) {
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
			cocciStream << "-lhs="<< fname <<"(arg1,arg2,arg3,arg4,arg5,arg6);\n";
			cocciStream << "+taskid = gettask(); // TRANSFORMED from version 0 ---\n";
			cocciStream << "+lhs=translate_task(taskid,arg1,arg2,arg3,arg4,arg5,arg6);\n";
			//cocciStream << "\n...>\n";
			cocciFptr << cocciStream.str();
		}
		else {
			string arg1=fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
			string arg2=fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
			string arg3=fArgs->get_traversalSuccessorByIndex(3)->unparseToString();

			cocciStream << "@@ \n";
			cocciStream << "type ty; \n";
			cocciStream << "typedef bool; \n";
			cocciStream << "identifier twoel; \n";
			cocciStream << "@@ \n";

			cocciStream << "ty twoel(...){ \n";
			cocciStream << "<... \n";

			cocciStream << "while(...){ \n";
			cocciStream << "... \n";
			cocciStream << fname<<"("<<fArgs->unparseToCompleteString() <<");   \n";
			cocciStream << "+// ADDED from version 2 -- \n";
			cocciStream << "+if (!is_task_real("<< arg3 <<", *schwmax, "<< arg1 <<","<< arg2 <<")) \n";
			cocciStream << "+     continue; // skip fake tasks \n";

			cocciStream << "... \n";
			cocciStream << "} \n";

			cocciStream << "...> \n";
			cocciStream << "} \n\n\n";


			cocciFptr << cocciStream.str();
		}
	}
	else if(version==1){
		string chunk1 = "gettask";
		string chunk2 = "translate_task";
		if(fname.compare(chunk1)==0){
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
			cocciStream << "+ //TASCELSEDREP int *map = new int[2 * 2 * nproc]; \n";
			cocciStream << "+ //TASCELSEDREP int *procs = new int[nproc]; \n";
			cocciStream << "+ \n";
			cocciStream << "+ me = GA_Nodeid(); \n";
			cocciStream << "+ \n";
			cocciStream << "+ // ADDED from version 1 --  \n";
			cocciStream << "+ imax = nbfn / ichunk;  \n";
			cocciStream << "+ tottasks = imax * imax * imax * imax; // n^4 total tasks \n";
			cocciStream << "+ \n";
			cocciStream << "... \n";
			cocciStream << "-taskid = "<< chunk1 <<"();\n";
			cocciStream << "-dotask = "<< chunk2 <<"(taskid,arg1,arg2,arg3,arg4,arg5,arg6);\n";
			cocciStream << "... \n";


			cocciStream << "-while(...){ \n";
			cocciStream << "+for (taskid = 0; taskid < tottasks; taskid++) { \n";
			cocciStream << "+ dotask = TASCELSEDREP + "<< chunk2 <<"(taskid,arg1,arg2,arg3,arg4,arg5,arg6); \n";
			cocciStream << "+ \n";
			cocciStream << "+if (!is_task_local(g_schwarz, arg1,arg2, me, map, procs)) \n";
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

			cocciStream << "-tid = "<< fname <<"();\n\n\n";


			cocciStream << "@@ \n";
			cocciStream << "type t; \n";
			cocciStream << "typedef bool; \n";
			cocciStream << "@@ \n";
			cocciStream << "t "<< fname <<"(...) { ... } \n\n";

			cocciStream << "+ bool is_task_local(int g_a, int *lo, int *hi, int me, int *map, int *procs) \n";
			cocciStream << "+ { \n";
			cocciStream << "+  int np = NGA_Locate_region(g_a, lo, hi, map, procs); \n";
			cocciStream << "+  return (procs[0] == me); \n";
			cocciStream << "+ } // is_task_local \n\n\n";

			cocciStream << "+ // ADDED from version 2 \n";
			cocciStream << "+ bool is_task_real(double s_ij[][ichunk], double schwmax, int lo[], int hi[]) \n";
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

		else if(fname.compare(chunk2)==0){
			cocciStream << "@@\n";
			cocciStream << "identifier lhs;\n";
			cocciStream << "expression arg0,arg1,arg2,arg3,arg4,arg5,arg6;\n";
			cocciStream << "type t;\n";
			cocciStream << "@@\n";
			cocciStream << "<...\n";
			cocciStream << "\n";
			cocciStream << "t lhs;\n";
			cocciStream << "...>\n\n";

			cocciStream << "-lhs="<< fname <<"(arg0,arg1,arg2,arg3,arg4,arg5,arg6);\n";

			cocciFptr << cocciStream.str();
		}
	}

}
