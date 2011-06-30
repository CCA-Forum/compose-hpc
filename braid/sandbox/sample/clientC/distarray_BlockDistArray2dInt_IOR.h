/*
 * File:          distarray_BlockDistArray2dInt_IOR.h
 * Symbol:        distarray.BlockDistArray2dInt-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Intermediate Object Representation for distarray.BlockDistArray2dInt
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_distarray_BlockDistArray2dInt_IOR_h
#define included_distarray_BlockDistArray2dInt_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "distarray.BlockDistArray2dInt" (version 0.1)
 */

struct distarray_BlockDistArray2dInt__array;
struct distarray_BlockDistArray2dInt__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Declare the method entry point vector.
 */

struct distarray_BlockDistArray2dInt__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 7 */
  void (*f__set_contracts)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 8 */
  void (*f__dump_stats)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 9 */
  void (*f__ctor)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 10 */
  void (*f__ctor2)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 11 */
  void (*f__dtor)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 12 */
  void (*f__load)(
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.17 */
  void (*f_addRef)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_deleteRef)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isType)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.17 */
  /* Methods introduced in distarray.BlockDistArray2dInt-v0.1 */
  void (*f_initArray)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t lo1,
    /* in */ int32_t hi1,
    /* in */ int32_t lo2,
    /* in */ int32_t hi2,
    /* in */ int32_t blk1,
    /* in */ int32_t blk2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  int32_t (*f_getDimension)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  int32_t (*f_getLower)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t dim,
    /* out */ struct sidl_BaseInterface__object **_ex);
  int32_t (*f_getHigher)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t dim,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getFromArray)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_setIntoArray)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t newVal,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method pre hooks entry point vector.
 */

struct distarray_BlockDistArray2dInt__pre_epv {
  void (*f_initArray_pre)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t lo1,
    /* in */ int32_t hi1,
    /* in */ int32_t lo2,
    /* in */ int32_t hi2,
    /* in */ int32_t blk1,
    /* in */ int32_t blk2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getDimension_pre)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getLower_pre)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t dim,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getHigher_pre)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t dim,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getFromArray_pre)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_setIntoArray_pre)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t newVal,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method post hooks entry point vector.
 */

struct distarray_BlockDistArray2dInt__post_epv {
  void (*f_initArray_post)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t lo1,
    /* in */ int32_t hi1,
    /* in */ int32_t lo2,
    /* in */ int32_t hi2,
    /* in */ int32_t blk1,
    /* in */ int32_t blk2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getDimension_post)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t _retval,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getLower_post)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t dim,
    /* in */ int32_t _retval,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getHigher_post)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t dim,
    /* in */ int32_t _retval,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_getFromArray_post)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_setIntoArray_post)(
    /* in */ struct distarray_BlockDistArray2dInt__object* self,
    /* in */ int32_t newVal,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Define the controls and statistics structure.
 */


struct distarray_BlockDistArray2dInt__cstats {
  sidl_bool use_hooks;
};

/*
 * Define the class object structure.
 */

struct distarray_BlockDistArray2dInt__object {
  struct sidl_BaseClass__object                d_sidl_baseclass;
  struct distarray_BlockDistArray2dInt__epv*   d_epv;
  struct distarray_BlockDistArray2dInt__cstats d_cstats;
  void*                                        d_data;
};

struct distarray_BlockDistArray2dInt__external {
  struct distarray_BlockDistArray2dInt__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct distarray_BlockDistArray2dInt__external*
distarray_BlockDistArray2dInt__externals(void);

extern struct distarray_BlockDistArray2dInt__object*
distarray_BlockDistArray2dInt__createObject(void* ddata,struct 
  sidl_BaseInterface__object ** _ex);

extern void distarray_BlockDistArray2dInt__init(
  struct distarray_BlockDistArray2dInt__object* self, void* ddata, struct 
    sidl_BaseInterface__object ** _ex);

extern void distarray_BlockDistArray2dInt__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
  struct distarray_BlockDistArray2dInt__epv 
    **s_arg_epv__distarray_blockdistarray2dint,
  struct distarray_BlockDistArray2dInt__epv 
    **s_arg_epv_hooks__distarray_blockdistarray2dint);

extern void distarray_BlockDistArray2dInt__fini(
  struct distarray_BlockDistArray2dInt__object* self, struct 
    sidl_BaseInterface__object ** _ex);

extern void distarray_BlockDistArray2dInt__IOR_version(int32_t *major, int32_t 
  *minor);

struct sidl_BaseInterface__object* 
  skel_distarray_BlockDistArray2dInt_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, struct sidl_BaseInterface__object * *_ex);
struct distarray_BlockDistArray2dInt__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
