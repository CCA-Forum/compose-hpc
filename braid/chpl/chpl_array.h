#ifndef _SIDLARRAY_H_
#define _SIDLARRAY_H_
#include <sidlArray.h>

// Chapel-compatible typedef
typedef struct sidl_array _sidl_array;
typedef _sidl_array* sidl_array;

// Macro definitions for Chapel-generated C code
#define sidlArrayElem1Set(array, ind1, val) \
  sidlArrayElem1(array,ind1) = val

#endif
