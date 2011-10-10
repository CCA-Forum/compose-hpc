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
}
