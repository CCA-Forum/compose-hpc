#ifndef BRAID_CHAPEL_UTIL_H
#define BRAID_CHAPEL_UTIL_H

#define GET_REF(anExpr) (&(anExpr))

#define MAKE_OPAQUE(inRef) (inRef)

/**
 * typedefs to allow chapel to compile the program
 */
typedef struct sidl_BaseInterface__object* sidl_BaseInterface__struct;
typedef struct hplsupport_BlockCyclicDistArray2dDouble__object* hplsupport_BlockCyclicDistArray2dDouble__struct;

#endif
