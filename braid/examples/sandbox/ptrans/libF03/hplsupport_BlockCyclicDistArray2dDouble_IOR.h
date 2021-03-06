/*
 * File:          hplsupport_BlockCyclicDistArray2dDouble_IOR.h
 * Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 7138  trunk)
 * Description:   Intermediate Object Representation for hplsupport.BlockCyclicDistArray2dDouble
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_IOR_h
#define included_hplsupport_BlockCyclicDistArray2dDouble_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifndef VAR_UNUSED
#ifdef __GNUC__
#define VAR_UNUSED __attribute__ ((__unused__))
#else
#define VAR_UNUSED
#endif /* __GNUC__ */
#endif /* VAR_UNUSED */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "hplsupport.BlockCyclicDistArray2dDouble" (version 0.1)
 */

struct hplsupport_BlockCyclicDistArray2dDouble__array;
struct hplsupport_BlockCyclicDistArray2dDouble__object;

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

struct hplsupport_BlockCyclicDistArray2dDouble__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 7 */
  void (*f__set_contracts)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 8 */
  void (*f__dump_stats)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 9 */
  void (*f__ctor)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 10 */
  void (*f__ctor2)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 11 */
  void (*f__dtor)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 12 */
  void (*f__load)(
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.17 */
  void (*f_addRef)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_deleteRef)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isType)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.17 */
  /* Methods introduced in hplsupport.BlockCyclicDistArray2dDouble-v0.1 */
  void (*f_initData)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ void* data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  double (*f_get)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_set)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ double newVal,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method pre hooks entry point vector.
 */

struct hplsupport_BlockCyclicDistArray2dDouble__pre_epv {
  void (*f_initData_pre)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ void* data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_get_pre)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_set_pre)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ double newVal,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method post hooks entry point vector.
 */

struct hplsupport_BlockCyclicDistArray2dDouble__post_epv {
  void (*f_initData_post)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ void* data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_get_post)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* in */ double _retval,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_set_post)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
    /* in */ double newVal,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Define the controls and statistics structure.
 */


struct hplsupport_BlockCyclicDistArray2dDouble__cstats {
  sidl_bool use_hooks;
};

/*
 * Define the class object structure.
 */

struct hplsupport_BlockCyclicDistArray2dDouble__object {
  struct sidl_BaseClass__object                          d_sidl_baseclass;
  struct hplsupport_BlockCyclicDistArray2dDouble__epv*   d_epv;
  struct hplsupport_BlockCyclicDistArray2dDouble__cstats d_cstats;
  void*                                                  d_data;
};

struct hplsupport_BlockCyclicDistArray2dDouble__external {
  struct hplsupport_BlockCyclicDistArray2dDouble__object*
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

const struct hplsupport_BlockCyclicDistArray2dDouble__external*
hplsupport_BlockCyclicDistArray2dDouble__externals(void);

extern struct hplsupport_BlockCyclicDistArray2dDouble__object*
hplsupport_BlockCyclicDistArray2dDouble__createObject(void* ddata,struct 
  sidl_BaseInterface__object ** _ex);

extern void hplsupport_BlockCyclicDistArray2dDouble__init(
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self, void* ddata, 
    struct sidl_BaseInterface__object ** _ex);

extern void hplsupport_BlockCyclicDistArray2dDouble__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
  struct hplsupport_BlockCyclicDistArray2dDouble__epv 
    **s_arg_epv__hplsupport_blockcyclicdistarray2ddouble,
  struct hplsupport_BlockCyclicDistArray2dDouble__epv 
    **s_arg_epv_hooks__hplsupport_blockcyclicdistarray2ddouble);

extern void hplsupport_BlockCyclicDistArray2dDouble__fini(
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self, struct 
    sidl_BaseInterface__object ** _ex);

extern void hplsupport_BlockCyclicDistArray2dDouble__IOR_version(int32_t *major,
  int32_t *minor);

struct sidl_BaseInterface__object* 
  skel_hplsupport_BlockCyclicDistArray2dDouble_fconnect_sidl_BaseInterface(
  const char* url, sidl_bool ar, struct sidl_BaseInterface__object * *_ex);
struct hplsupport_BlockCyclicDistArray2dDouble__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
