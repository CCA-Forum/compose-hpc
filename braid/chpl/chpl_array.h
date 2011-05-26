#ifndef _SIDLARRAY_H_
#define _SIDLARRAY_H_
#include <sidlArray.h>

// Macro definitions for Chapel-generated C code
#define sidlArrayElem1Set(array, ind1, val) \
  sidlArrayElem1(array,ind1) = val

#endif
