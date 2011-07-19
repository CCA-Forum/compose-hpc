#ifndef __BLAS_VECTORUTILS_IOR_H__
#define __BLAS_VECTORUTILS_IOR_H__
#include <blas.h>
#include <sidl.h>
#include <sidl_BaseInterface_IOR.h>
#include <stdint.h>
#include <chpl_sidl_array.h>
struct blas_VectorUtils__cstats {
  sidl_bool use_hooks;
};

struct blas_VectorUtils__object {
  struct sidl_BaseClass__object d_sidl_baseclass;
  struct blas_VectorUtils__epv* d_epv;
  struct blas_VectorUtils__cstats d_cstats;
  void* d_data;
};

struct blas_VectorUtils__external {
  struct blas_VectorUtils__object* (*createObject)( void** ddata, struct 
    sidl_BaseInterface__object** ex);
  struct blas_VectorUtils__sepv* (*getStaticEPV)();
  struct sidl_BaseClass__epv (*getSuperEPV)();
  int d_ior_major_version;
  int d_ior_minor_version;
};

struct blas_VectorUtils__epv {
  void (*f__cast)( struct blas_VectorUtils__object* self, const char* name, 
    struct sidl_BaseInterface__object** ex);
  void (*f__delete)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  void (*f__exec)( struct blas_VectorUtils__object* self, const char* methodName, 
    void FIXMEinArgs, void FIXMEoutArgs, struct sidl_BaseInterface__object** ex);
  const char* (*f__getURL)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  void (*f__raddRef)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  int (*f__isRemote)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  void (*f__set_hooks)( struct blas_VectorUtils__object* self, int enable, struct 
    sidl_BaseInterface__object** ex);
  void (*f__set_contracts)( struct blas_VectorUtils__object* self, int enable, 
    const char* enfFilename, int resetCounters, struct sidl_BaseInterface__object** 
    ex);
  void (*f__dump_stats)( struct blas_VectorUtils__object* self, const char* 
    filename, const char* prefix, struct sidl_BaseInterface__object** ex);
  void (*f__ctor)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  void (*f__ctor2)( struct blas_VectorUtils__object* self, void private_data, 
    struct sidl_BaseInterface__object** ex);
  void (*f__dtor)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  void (*f__load)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  void (*f_addRef)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  void (*f_deleteRef)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object** ex);
  int (*f_isSame)( struct blas_VectorUtils__object* self, struct 
    sidl_BaseInterface__object iobj, struct sidl_BaseInterface__object** ex);
  int (*f_isType)( struct blas_VectorUtils__object* self, const char* type, 
    struct sidl_BaseInterface__object** ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)( struct 
    blas_VectorUtils__object* self, struct sidl_BaseInterface__object** ex);
};

struct blas_VectorUtils__sepv {
  void (*f_setHooks_static)( int enable, struct sidl_BaseInterface__object** ex);
  void (*f_set_contracts_static)( int enable, const char* enfFilename, int 
    resetCounters, struct sidl_BaseInterface__object** ex);
  void (*f_dump_stats_static)( const char* filename, const char* prefix, struct 
    sidl_BaseInterface__object** ex);
  void (*f_helper_daxpy)( int n, double alpha, sidl_double__array X, 
    sidl_double__array Y, struct sidl_BaseInterface__object** ex);
};

/**
 * INIT: initialize a new instance of the class object.
 */
void blas_VectorUtils__init( struct blas_VectorUtils__object* self, void* data, struct sidl_BaseInterface__object** ex);
/**
 * FINI: deallocate a class instance (destructor).
 */
void blas_VectorUtils__fini( struct blas_VectorUtils__object* self, struct sidl_BaseInterface__object** ex);

#endif
