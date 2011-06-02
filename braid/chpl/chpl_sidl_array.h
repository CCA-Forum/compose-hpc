#ifndef _CHPL_SIDL_ARRAY_H_
#define _CHPL_SIDL_ARRAY_H_

#include <sidlArray.h>
#include <sidl_bool_IOR.h>        
#include <sidl_char_IOR.h>        
#include <sidl_dcomplex_IOR.h>    
#include <sidl_double_IOR.h>      
#include <sidl_fcomplex_IOR.h>    
#include <sidl_float_IOR.h>       
#include <sidl_int_IOR.h>         
#include <sidl_long_IOR.h>        
#include <sidl_opaque_IOR.h>      
#include <sidl_string_IOR.h>      
#include <sidl_BaseInterface_IOR.h>

struct sidl_string__array {
  struct sidl__array   d_metadata;
  char * *d_firstElement;
};


// Chapel-compatible typedef
#define CHAPEL_TYPEDEF(T) \
  typedef struct T _##T; \
  typedef _##T* T;

CHAPEL_TYPEDEF(sidl__array)
CHAPEL_TYPEDEF(sidl_bool__array)        
CHAPEL_TYPEDEF(sidl_char__array)        
CHAPEL_TYPEDEF(sidl_dcomplex__array)    
CHAPEL_TYPEDEF(sidl_double__array)      
CHAPEL_TYPEDEF(sidl_fcomplex__array)    
CHAPEL_TYPEDEF(sidl_float__array)       
CHAPEL_TYPEDEF(sidl_int__array)         
CHAPEL_TYPEDEF(sidl_long__array)        
CHAPEL_TYPEDEF(sidl_opaque__array)      
CHAPEL_TYPEDEF(sidl_string__array)      
CHAPEL_TYPEDEF(sidl_BaseInterface__array)

// Macro definitions for Chapel-generated C code
#define sidlArrayElem1Set(array, ind1, val) \
  sidlArrayElem1(array,ind1) = val

// Identity function. We use it to cast * -> opaque
#define int_ptr(A) A
#endif
