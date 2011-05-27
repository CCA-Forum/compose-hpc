#ifndef _CHPL_SIDL_ARRAY_H_
#define _CHPL_SIDL_ARRAY_H_
#include <sidlArray.h>

// Chapel-compatible typedef
typedef struct sidl__array _sidl__array;
typedef _sidl__array* sidl__array;

// Macro definitions for Chapel-generated C code
#define sidlArrayElem1Set(array, ind1, val) \
  sidlArrayElem1(array,ind1) = val

#endif
