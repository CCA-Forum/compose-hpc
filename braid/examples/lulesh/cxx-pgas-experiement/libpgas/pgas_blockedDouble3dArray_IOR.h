#ifndef included_pgas_blockedDouble3dArray_IOR_h
#define included_pgas_blockedDouble3dArray_IOR_h
#include <pgas.h>
#include <sidl.h>
#include <sidl_BaseClass_IOR.h>
#include <stdint.h>
#include <chpl_sidl_array.h>
#include <chpltypes.h>
struct pgas_blockedDouble3dArray__array;
struct pgas_blockedDouble3dArray__object;
struct sidl_BaseClass__array;
struct sidl_BaseClass__object;
struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Define invariant clause data for interface contract enforcement.
 */

static VAR_UNUSED struct pgas_blockedDouble3dArray__inv_desc{
  int    inv_complexity;
  double inv_exec_time;
} s_ior_pgas_blockedDouble3dArray_inv = {
  0, 0.0,
};

/*
 * Define method description data for interface contract enforcement.
 */

static const int32_t s_IOR_PGAS_BLOCKEDDOUBLE3DARRAY_MIN = 0;
static const int32_t s_IOR_PGAS_BLOCKEDDOUBLE3DARRAY_ALLOCATE = 0;
static const int32_t s_IOR_PGAS_BLOCKEDDOUBLE3DARRAY_GET = 1;
static const int32_t s_IOR_PGAS_BLOCKEDDOUBLE3DARRAY_SET = 2;
static const int32_t s_IOR_PGAS_BLOCKEDDOUBLE3DARRAY_MAX = 3;

static VAR_UNUSED struct pgas_blockedDouble3dArray__method_desc{
  const char* name;
  sidl_bool   is_static;
  long        est_interval;
  int         pre_complexity;
  int         post_complexity;
  double      meth_exec_time;
  double      pre_exec_time;
  double      post_exec_time;
} s_ior_pgas_blockedDouble3dArray_method[] = {

{"allocate", 1, 0, 0, 0, 0.0, 0.0, 0.0},
{"get", 1, 0, 0, 0, 0.0, 0.0, 0.0},
{"set", 1, 0, 0, 0, 0.0, 0.0, 0.0},
};

/* static structure options */
static const int32_t s_SEPV_PGAS_BLOCKEDDOUBLE3DARRAY_BASE      = 0;
static const int32_t s_SEPV_PGAS_BLOCKEDDOUBLE3DARRAY_CONTRACTS = 1;
static const int32_t s_SEPV_PGAS_BLOCKEDDOUBLE3DARRAY_HOOKS     = 2;


struct pgas_blockedDouble3dArray__cstats {
  sidl_bool use_hooks;
};

struct pgas_blockedDouble3dArray__object {
  struct sidl_BaseClass__object d_sidl_baseclass;
  struct pgas_blockedDouble3dArray__epv* d_epv;
  struct pgas_blockedDouble3dArray__cstats d_cstats;
  void* d_data;
};

struct pgas_blockedDouble3dArray__external {
  struct pgas_blockedDouble3dArray__object* (*createObject)( void** ddata,   struct sidl_BaseInterface__object** _ex);
  struct sidl_BaseClass__epv* (*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

struct pgas_blockedDouble3dArray__epv {
  void* (*f__cast)( struct pgas_blockedDouble3dArray__object* self,   const char* name,   struct sidl_BaseInterface__object** _ex);
  void (*f__delete)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f__exec)( struct pgas_blockedDouble3dArray__object* self,   const char* methodName,   struct sidl_rmi_Call__object* inArgs,   struct sidl_rmi_Return__object* outArgs,   struct sidl_BaseInterface__object** _ex);
  const char* (*f__getURL)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f__raddRef)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  sidl_bool (*f__isRemote)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f__set_hooks)( struct pgas_blockedDouble3dArray__object* self,   sidl_bool enable,   struct sidl_BaseInterface__object** _ex);
  void (*f__set_contracts)( struct pgas_blockedDouble3dArray__object* self,   sidl_bool enable,   const char* enfFilename,   sidl_bool resetCounters,   struct sidl_BaseInterface__object** _ex);
  void (*f__dump_stats)( struct pgas_blockedDouble3dArray__object* self,   const char* filename,   const char* prefix,   struct sidl_BaseInterface__object** _ex);
  void (*f__ctor)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f__ctor2)( struct pgas_blockedDouble3dArray__object* self,   void* private_data,   struct sidl_BaseInterface__object** _ex);
  void (*f__dtor)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f__load)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f_addRef)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f_deleteRef)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  sidl_bool (*f_isSame)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object* iobj,   struct sidl_BaseInterface__object** _ex);
  sidl_bool (*f_isType)( struct pgas_blockedDouble3dArray__object* self,   const char* name,   struct sidl_BaseInterface__object** _ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)( struct pgas_blockedDouble3dArray__object* self,   struct sidl_BaseInterface__object** _ex);
  void (*f_allocate)( struct pgas_blockedDouble3dArray__object* self,   int size,   struct sidl_BaseInterface__object** _ex);
  double (*f_get)( struct pgas_blockedDouble3dArray__object* self,   int idx1,   int idx2,   int idx3,   struct sidl_BaseInterface__object** _ex);
  void (*f_set)( struct pgas_blockedDouble3dArray__object* self,   int idx1,   int idx2,   int idx3,   double val,   struct sidl_BaseInterface__object** _ex);
};

struct pgas_blockedDouble3dArray__pre_epv {
  void (*f_allocate_pre)( struct pgas_blockedDouble3dArray__object* self,   int size,   struct sidl_BaseInterface__object** _ex);
  double (*f_get_pre)( struct pgas_blockedDouble3dArray__object* self,   int idx1,   int idx2,   int idx3,   struct sidl_BaseInterface__object** _ex);
  void (*f_set_pre)( struct pgas_blockedDouble3dArray__object* self,   int idx1,   int idx2,   int idx3,   double val,   struct sidl_BaseInterface__object** _ex);
};

struct pgas_blockedDouble3dArray__post_epv {
  void (*f_allocate_post)( struct pgas_blockedDouble3dArray__object* self,   int size,   struct sidl_BaseInterface__object** _ex);
  double (*f_get_post)( struct pgas_blockedDouble3dArray__object* self,   int idx1,   int idx2,   int idx3,   struct sidl_BaseInterface__object** _ex);
  void (*f_set_post)( struct pgas_blockedDouble3dArray__object* self,   int idx1,   int idx2,   int idx3,   double val,   struct sidl_BaseInterface__object** _ex);
};

/**
 * INIT: initialize a new instance of the class object.
 */
void pgas_blockedDouble3dArray__init( struct pgas_blockedDouble3dArray__object* self, void* ddata, struct sidl_BaseInterface__object** _ex);
/**
 * FINI: deallocate a class instance (destructor).
 */
void pgas_blockedDouble3dArray__fini( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex);
struct pgas_blockedDouble3dArray__object* 
  pgas_blockedDouble3dArray__createObject(void* ddata, struct 
  sidl_BaseInterface__object ** _ex);
#endif

