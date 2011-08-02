/*
 * File:          hpcc_HighPerformanceLinpack_IOR.h
 * Symbol:        hpcc.HighPerformanceLinpack-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Intermediate Object Representation for hpcc.HighPerformanceLinpack
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_hpcc_HighPerformanceLinpack_IOR_h
#define included_hpcc_HighPerformanceLinpack_IOR_h

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
 * Symbol "hpcc.HighPerformanceLinpack" (version 0.1)
 */

struct hpcc_HighPerformanceLinpack__array;
struct hpcc_HighPerformanceLinpack__object;
struct hpcc_HighPerformanceLinpack__sepv;

/*
 * Forward references for external classes and interfaces.
 */

struct hplsupport_BlockCyclicDistArray2dDouble__array;
struct hplsupport_BlockCyclicDistArray2dDouble__object;
struct hplsupport_SimpleArray1dInt__array;
struct hplsupport_SimpleArray1dInt__object;
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
 * Declare the static method entry point vector.
 */

struct hpcc_HighPerformanceLinpack__sepv {
  /* Implicit builtin methods */
  /* 0 */
  /* 1 */
  /* 2 */
  /* 3 */
  /* 4 */
  /* 5 */
  /* 6 */
  void (*f__set_hooks_static)(
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 7 */
  void (*f__set_contracts_static)(
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 8 */
  void (*f__dump_stats_static)(
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 9 */
  /* 10 */
  /* 11 */
  /* 12 */
  /* Methods introduced in sidl.BaseInterface-v0.9.17 */
  /* Methods introduced in sidl.BaseClass-v0.9.17 */
  /* Methods introduced in hpcc.HighPerformanceLinpack-v0.1 */
  void (*f_panelSolveCompute)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* ab,
    /* in */ struct hplsupport_SimpleArray1dInt__object* piv,
    /* in */ int32_t abStart1,
    /* in */ int32_t abEnd1,
    /* in */ int32_t abStart2,
    /* in */ int32_t abEnd2,
    /* in */ int32_t start1,
    /* in */ int32_t end1,
    /* in */ int32_t start2,
    /* in */ int32_t end2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the method entry point vector.
 */

struct hpcc_HighPerformanceLinpack__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 7 */
  void (*f__set_contracts)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 8 */
  void (*f__dump_stats)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 9 */
  void (*f__ctor)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 10 */
  void (*f__ctor2)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 11 */
  void (*f__dtor)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* 12 */
  void (*f__load)(
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseInterface-v0.9.17 */
  void (*f_addRef)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  void (*f_deleteRef)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex);
  sidl_bool (*f_isType)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct hpcc_HighPerformanceLinpack__object* self,
    /* out */ struct sidl_BaseInterface__object **_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.17 */
  /* Methods introduced in hpcc.HighPerformanceLinpack-v0.1 */
};

/*
 * Declare the method pre hooks entry point vector.
 */

struct hpcc_HighPerformanceLinpack__pre_epv {
  /* Avoid empty struct */
  char d_not_empty;
};

/*
 * Declare the method post hooks entry point vector.
 */

struct hpcc_HighPerformanceLinpack__post_epv {
  /* Avoid empty struct */
  char d_not_empty;
};

/*
 * Declare the static method pre hooks entry point vector.
 */

struct hpcc_HighPerformanceLinpack__pre_sepv {
  void (*f_panelSolveCompute_pre)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* ab,
    /* in */ struct hplsupport_SimpleArray1dInt__object* piv,
    /* in */ int32_t abStart1,
    /* in */ int32_t abEnd1,
    /* in */ int32_t abStart2,
    /* in */ int32_t abEnd2,
    /* in */ int32_t start1,
    /* in */ int32_t end1,
    /* in */ int32_t start2,
    /* in */ int32_t end2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Declare the static method post hooks entry point vector.
 */

struct hpcc_HighPerformanceLinpack__post_sepv {
  void (*f_panelSolveCompute_post)(
    /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object* ab,
    /* in */ struct hplsupport_SimpleArray1dInt__object* piv,
    /* in */ int32_t abStart1,
    /* in */ int32_t abEnd1,
    /* in */ int32_t abStart2,
    /* in */ int32_t abEnd2,
    /* in */ int32_t start1,
    /* in */ int32_t end1,
    /* in */ int32_t start2,
    /* in */ int32_t end2,
    /* out */ struct sidl_BaseInterface__object **_ex);
};

/*
 * Define the controls and statistics structure.
 */


struct hpcc_HighPerformanceLinpack__cstats {
  sidl_bool use_hooks;
};

/*
 * Define the class object structure.
 */

struct hpcc_HighPerformanceLinpack__object {
  struct sidl_BaseClass__object              d_sidl_baseclass;
  struct hpcc_HighPerformanceLinpack__epv*   d_epv;
  struct hpcc_HighPerformanceLinpack__cstats d_cstats;
  void*                                      d_data;
};

struct hpcc_HighPerformanceLinpack__external {
  struct hpcc_HighPerformanceLinpack__object*
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct hpcc_HighPerformanceLinpack__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct hpcc_HighPerformanceLinpack__external*
hpcc_HighPerformanceLinpack__externals(void);

extern struct hpcc_HighPerformanceLinpack__object*
hpcc_HighPerformanceLinpack__createObject(void* ddata,struct 
  sidl_BaseInterface__object ** _ex);

extern struct hpcc_HighPerformanceLinpack__sepv*
hpcc_HighPerformanceLinpack__getStaticEPV(void);


extern struct hpcc_HighPerformanceLinpack__sepv*
hpcc_HighPerformanceLinpack__getTypeStaticEPV(int type);

extern void hpcc_HighPerformanceLinpack__init(
  struct hpcc_HighPerformanceLinpack__object* self, void* ddata, struct 
    sidl_BaseInterface__object ** _ex);

extern void hpcc_HighPerformanceLinpack__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
  struct hpcc_HighPerformanceLinpack__epv 
    **s_arg_epv__hpcc_highperformancelinpack,
  struct hpcc_HighPerformanceLinpack__epv 
    **s_arg_epv_hooks__hpcc_highperformancelinpack);

extern void hpcc_HighPerformanceLinpack__fini(
  struct hpcc_HighPerformanceLinpack__object* self, struct 
    sidl_BaseInterface__object ** _ex);

extern void hpcc_HighPerformanceLinpack__IOR_version(int32_t *major, int32_t 
  *minor);

/*
 * Define static structure options.
 */

static const int s_SEPV_HPCC_HIGHPERFORMANCELINPACK_BASE = 0;
static const int s_SEPV_HPCC_HIGHPERFORMANCELINPACK_CONTRACTS = 1;
static const int s_SEPV_HPCC_HIGHPERFORMANCELINPACK_HOOKS = 2;

struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  skel_hpcc_HighPerformanceLinpack_fconnect_hplsupport_BlockCyclicDistArray2dDouble
  (const char* url, sidl_bool ar, struct sidl_BaseInterface__object * *_ex);
struct hplsupport_SimpleArray1dInt__object* 
  skel_hpcc_HighPerformanceLinpack_fconnect_hplsupport_SimpleArray1dInt(const 
  char* url, sidl_bool ar, struct sidl_BaseInterface__object * *_ex);
struct sidl_BaseInterface__object* 
  skel_hpcc_HighPerformanceLinpack_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, struct sidl_BaseInterface__object * *_ex);
struct hpcc_HighPerformanceLinpack__remote{
  int d_refcount;
  struct sidl_rmi_InstanceHandle__object *d_ih;
};

#ifdef __cplusplus
}
#endif
#endif
