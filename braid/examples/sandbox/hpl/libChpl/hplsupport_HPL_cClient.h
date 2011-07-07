
#ifndef HPLSUPPORT_HPL_CLIENT_H
#define HPLSUPPORT_HPL_CLIENT_H

#include <stdio.h>

void panelSolveNative(void* abData, void* pivData,
		/* abLimits*/ int abStart1, int abEnd1, int abStart2, int abEnd2,
		/*panel domain*/ int start1, int end1, int start2, int end2);

#endif


