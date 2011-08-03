#include <stdlib.h>
#include "d_stub.h"

/* stub file for rose to put output in and to test cocinelle rewrites on. */

#define MULT(a,b) (a*b)

int dummy_stub = 1;

static int dummy_stub2=2;

extern int dummy_stub3;

static struct DType { 
	int x;
	long y;
} DType0 = { -1, 17L };

int simpleton(struct dummy_insert *argadded,int x)
{
	if (argadded != (struct dummy_insert *)4096) {
		exit(2);
	}
	static int dummy_first;
	return MULT(MULT(dummy_first,x), dummy_stub2);
}

int fred(struct dummy_insert *argadded,int x)
{
	if (argadded != (struct dummy_insert *)4096) {
		exit(2);
	}
	return simpleton(((struct dummy_insert *)4096), 3);
}
