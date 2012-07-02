
/*
 * Begin: RMI includes
 */

#include "sidl_rmi_InstanceHandle.h"
#include "sidl_rmi_InstanceRegistry.h"
#include "sidl_rmi_ServerRegistry.h"
#include "sidl_rmi_Call.h"
#include "sidl_rmi_Return.h"
#include "sidl_exec_err.h"
#include "sidl_PreViolation.h"
#include "sidl_NotImplementedException.h"
#include <stdio.h>
/*
 * End: RMI includes
 */

#include "sidl_Exception.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>
#if TIME_WITH_SYS_TIME
#  include <sys/time.h>
#  include <time.h>
#else
#  if HAVE_SYS_TIME_H
#    include <sys/time.h>
#  else
#    include <time.h>
#  endif
#endif

#include "sidlAsserts.h"
#include "sidl_Enforcer.h"
/* #define SIDL_CONTRACTS_DEBUG 1 */

#define SIDL_NO_DISPATCH_ON_VIOLATION 1
#define SIDL_SIM_TRACE 1

#ifndef included_sidlOps_h
#include "sidlOps.h"
#endif

#include "pgas_blockedDoubleArray_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t pgas_blockedDoubleArray__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &pgas_blockedDoubleArray__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &pgas_blockedDoubleArray__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &pgas_blockedDoubleArray__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

#define RESETCD(MD) { \
  if (MD) { (MD)->est_interval = sidl_Enforcer_getEstimatesInterval(); } \
}
#ifdef SIDL_SIM_TRACE
#define TRACE(CN, MD, MID, PRC, POC, INC, MT, PRT, POT, IT1, IT2) { \
  if (MD) { sidl_Enforcer_logTrace(CN, (MD)->name, MID, PRC, POC, INC, MT, PRT, POT, IT1, IT2); } \
}
#else /* !SIDL_SIM_TRACE */
#define TRACE(CN, MD, MID, PRC, POC, INC, MT, PRT, POT, IT1, IT2)
#endif /* SIDL_SIM_TRACE */


/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 2;
static const int32_t s_IOR_MINOR_VERSION = 0;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo  = NULL;

/*
 * Static variable to make sure _load called no more than once.
 */

static int s_load_called = 0;

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_static_initialized = 0;

static struct pgas_blockedDoubleArray__epv s_my_epv__pgas_blockeddoublearray = { 0 };
static struct pgas_blockedDoubleArray__epv s_my_epv_contracts__pgas_blockeddoublearray = { 0 };
static struct pgas_blockedDoubleArray__epv s_my_epv_hooks__pgas_blockeddoublearray = { 0 };

static struct pgas_blockedDoubleArray__epv  s_my_epv__pgas_blockeddoublearray;
static struct pgas_blockedDoubleArray__epv  s_my_pre_epv_hooks__pgas_blockeddoublearray;
static struct pgas_blockedDoubleArray__epv*  s_par_epv__pgas_blockeddoublearray;
static struct pgas_blockedDoubleArray_pre__epv*  s_par_epv_hooks__pgas_blockeddoublearray;

static struct sidl_BaseClass__epv  s_my_epv__sidl_baseclass;
static struct sidl_BaseClass__epv*  s_par_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_my_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv*  s_par_epv__sidl_baseinterface;

static struct pgas_blockedDoubleArray__pre_epv s_preEPV;
static struct pgas_blockedDoubleArray__post_epv s_postEPV;


/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif


extern void pgas_blockedDoubleArray__set_epv(
  struct pgas_blockedDoubleArray__epv* epv,
    struct pgas_blockedDoubleArray__pre_epv* pre_epv,
    struct pgas_blockedDoubleArray__post_epv* post_epv);


extern void pgas_blockedDoubleArray__call_load(void);
#ifdef __cplusplus
}
#endif

static void
pgas_blockedDoubleArray_addRef__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}
static void
pgas_blockedDoubleArray_deleteRef__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}
static void
pgas_blockedDoubleArray_isSame__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}
static void
pgas_blockedDoubleArray_isType__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}
static void
pgas_blockedDoubleArray_getClassInfo__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}
static void
pgas_blockedDoubleArray_allocate__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}
static void
pgas_blockedDoubleArray_get__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}
static void
pgas_blockedDoubleArray_set__exec(
        struct pgas_blockedDoubleArray__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object** ex) {
}

/*
 * CHECKS: Enable/disable contract enforcement.
 */

static void ior_pgas_blockedDoubleArray__set_contracts(
  struct pgas_blockedDoubleArray__object* self,
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex)
{
  *_ex  = NULL;
  /* empty because contract checks not needed */
}

/*
 * DUMP: Dump interface contract enforcement statistics.
 */

static void ior_pgas_blockedDoubleArray__dump_stats(
  struct pgas_blockedDoubleArray__object* self,
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex)
{
  *_ex = NULL;
  /* empty because contract checks not needed */
}

/* CAST: dynamic type casting support. */
static void* ior_pgas_blockedDoubleArray__cast(
  struct pgas_blockedDoubleArray__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int cmp;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
    cmp = strcmp(name, "sidl.BaseClass");
  if (!cmp) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = ((struct sidl_BaseClass__object*)self);
    return cast;
  }
  else if (cmp < 0) {
    cmp = strcmp(name, "pgas.blockedDoubleArray");
    if (!cmp) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct pgas_blockedDoubleArray__object*)self);
      return cast;
    }
  }
  else if (cmp > 0) {
    cmp = strcmp(name, "sidl.BaseInterface");
    if (!cmp) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
  }
  return cast;
  EXIT:
  return NULL;
}


/*
 * HOOKS: Enable/disable hooks.
 */

static void ior_pgas_blockedDoubleArray__set_hooks(
  struct pgas_blockedDoubleArray__object* self,
  sidl_bool enable, struct sidl_BaseInterface__object **_ex )
{
  *_ex  = NULL;
  /*
   * Nothing else to do since hook methods not generated.
   */

}

/*
 * HOOKS: Enable/disable static hooks.
 */

static void ior_pgas_blockedDoubleArray__set_hooks_static(
  sidl_bool enable, struct sidl_BaseInterface__object **_ex )
{
  *_ex  = NULL;
  /*
   * Nothing else to do since hook methods not generated.
   */

}


/*
 * DUMP: Dump static interface contract enforcement statistics.
 */

static void ior_vect_Utils__dump_stats_static(
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex)
{
  *_ex = NULL;
  /* empty since there are no static contracts */
}

/*
 * CHECKS: Enable/disable static contract enforcement.
 */

static void ior_vect_Utils__set_contracts_static(
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex)
{
  *_ex  = NULL;
  /* empty since there are no static contracts */
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_pgas_blockedDoubleArray__delete(
  struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object **_ex)
{
  *_ex  = NULL; /* default to no exception */
  pgas_blockedDoubleArray__fini(self, _ex);
  memset((void*)self, 0, sizeof(struct pgas_blockedDoubleArray__object));
  free((void*) self);
}

static const char*
ior_pgas_blockedDoubleArray__getURL(
  struct pgas_blockedDoubleArray__object* self,
  struct sidl_BaseInterface__object **_ex)
{
  char* ret  = NULL;
  char* objid = sidl_rmi_InstanceRegistry_getInstanceByClass((sidl_BaseClass)self, _ex); SIDL_CHECK(*_ex);
  if (!objid) {
    objid = sidl_rmi_InstanceRegistry_registerInstance((sidl_BaseClass)self, _ex); SIDL_CHECK(*_ex);
  }
#ifdef WITH_RMI

  ret = sidl_rmi_ServerRegistry_getServerURL(objid, _ex); SIDL_CHECK(*_ex);

#else

  ret = objid;

#endif /*WITH_RMI*/
  return ret;
  EXIT:
  return NULL;
}
static void
ior_pgas_blockedDoubleArray__raddRef(
    struct pgas_blockedDoubleArray__object* self, sidl_BaseInterface* _ex) {
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}

static sidl_bool
ior_pgas_blockedDoubleArray__isRemote(
    struct pgas_blockedDoubleArray__object* self, sidl_BaseInterface* _ex) {
  *_ex  = NULL; /* default to no exception */
  return FALSE;
}

struct pgas_blockedDoubleArray__method {
  const char *d_name;
  void (*d_func)(struct pgas_blockedDoubleArray__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
};

static void
ior_pgas_blockedDoubleArray__exec(
  struct pgas_blockedDoubleArray__object* self,
  const char* methodName,
  struct sidl_rmi_Call__object* inArgs,
  struct sidl_rmi_Return__object* outArgs,
  struct sidl_BaseInterface__object **_ex )
{
  static const struct pgas_blockedDoubleArray__method  s_methods[] = {
    { "addRef", pgas_blockedDoubleArray_addRef__exec },
    { "deleteRef", pgas_blockedDoubleArray_deleteRef__exec },
    { "isSame", pgas_blockedDoubleArray_isSame__exec },
    { "isType", pgas_blockedDoubleArray_isType__exec },
    { "getClassInfo", pgas_blockedDoubleArray_getClassInfo__exec },
    { "allocate", pgas_blockedDoubleArray_allocate__exec },
    { "get", pgas_blockedDoubleArray_get__exec },
    { "set", pgas_blockedDoubleArray_set__exec },
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct pgas_blockedDoubleArray__method);
  *_ex  = NULL; /* default to no exception */

  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs, _ex); SIDL_CHECK(*_ex);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
  SIDL_THROW(*_ex, sidl_PreViolation, "method name not found");
  EXIT:
  return;
}


/* no type_static_epv since there are no static contracts */
  /* no check_* stubs since there are no contracts */

static void ior_pgas_blockedDoubleArray__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    s_load_called=1;
    pgas_blockedDoubleArray__call_load();

  }
}


// EPV: create method entry point vector (EPV) structure.
static void pgas_blockedDoubleArray__init_epv(void)
{
  // assert( " + IOR.getHaveLockStaticGlobalsMacroName() + " );
  struct pgas_blockedDoubleArray__epv*  epv = &s_my_epv__pgas_blockeddoublearray;
  struct pgas_blockedDoubleArray__epv*  e0 = &s_my_epv__pgas_blockeddoublearray;
  struct sidl_BaseClass__epv*           e1 = &s_my_epv__sidl_baseclass;

  struct sidl_BaseClass__epv*        s1 = NULL;

  // Get my parent's EPVs so I can start with their functions.
  sidl_BaseClass__getEPVs(
    &s_par_epv__sidl_baseinterface,
    &s_par_epv__sidl_baseclass
  );


  // Alias the static epvs to some handy small names.
  s1 = s_par_epv__sidl_baseclass;

  epv->f__cast           = ior_pgas_blockedDoubleArray__cast;
  epv->f__delete         = ior_pgas_blockedDoubleArray__delete;
  epv->f__exec           = ior_pgas_blockedDoubleArray__exec;
  epv->f__getURL         = ior_pgas_blockedDoubleArray__getURL;
  epv->f__raddRef        = ior_pgas_blockedDoubleArray__raddRef;
  epv->f__isRemote       = ior_pgas_blockedDoubleArray__isRemote;
  epv->f__set_hooks      = ior_pgas_blockedDoubleArray__set_hooks;
  epv->f__set_contracts  = ior_pgas_blockedDoubleArray__set_contracts;
  epv->f__dump_stats     = ior_pgas_blockedDoubleArray__dump_stats;
  epv->f__ctor           = NULL;
  epv->f__ctor2          = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = (void (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object **))s1->f_addRef;
  epv->f_deleteRef       = (void (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object **))s1->f_deleteRef;
  epv->f_isSame          = (sidl_bool (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object*, struct sidl_BaseInterface__object **))s1->f_isSame;
  epv->f_isType          = (sidl_bool (*)(struct pgas_blockedDoubleArray__object*, const char*, struct sidl_BaseInterface__object **))s1->f_isType;
  epv->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object **))s1->f_getClassInfo;
  epv->f_allocate        = NULL;
  epv->f_get             = NULL;
  epv->f_set             = NULL;

  pgas_blockedDoubleArray__set_epv(epv, &s_preEPV, &s_postEPV);

  // Override function pointers for pgas_blockedDoubleArray with mine, as needed.
  e0->f__cast           = (void* (*)(struct pgas_blockedDoubleArray__object*,const char*, struct sidl_BaseInterface__object**))epv->f__cast;
  e0->f__delete         = (void* (*)(struct pgas_blockedDoubleArray__object*,struct sidl_BaseInterface__object**))epv->f__delete;
  e0->f__getURL         = (char* (*)(struct pgas_blockedDoubleArray__object*,struct sidl_BaseInterface__object**))epv->f__getURL;
  e0->f__raddRef        = (void (*)(struct pgas_blockedDoubleArray__object*,struct sidl_BaseInterface__object**))epv->f__raddRef;
  e0->f__isRemote       = (sidl_bool* (*)(struct pgas_blockedDoubleArray__object*,struct sidl_BaseInterface__object**))epv->f__isRemote;
  e0->f__exec           = (void (*)(struct pgas_blockedDoubleArray__object*,const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct sidl_BaseInterface__object**))epv->f__exec;
  e0->f_addRef          = (void (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object **))epv->f_addRef;
  e0->f_deleteRef       = (void (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object **))epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object*, struct sidl_BaseInterface__object **))epv->f_isSame;
  e0->f_isType          = (sidl_bool (*)(struct pgas_blockedDoubleArray__object*, const char*, struct sidl_BaseInterface__object **))epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(struct pgas_blockedDoubleArray__object*, struct sidl_BaseInterface__object **))epv->f_getClassInfo;
  e0->f_allocate        = (void (*)(struct pgas_blockedDoubleArray__object*, int, struct sidl_BaseInterface__object **))epv->f_allocate;
  e0->f_get             = (double (*)(struct pgas_blockedDoubleArray__object*, int, struct sidl_BaseInterface__object **))epv->f_get;
  e0->f_set             = (void (*)(struct pgas_blockedDoubleArray__object*, int, double, struct sidl_BaseInterface__object **))epv->f_set;



  // Override function pointers for sidl_BaseClass with mine, as needed.
  e1->f__cast           = (void* (*)(struct sidl_BaseClass__object*,const char*, struct sidl_BaseInterface__object**))epv->f__cast;
  e1->f__delete         = (void* (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object**))epv->f__delete;
  e1->f__getURL         = (char* (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object**))epv->f__getURL;
  e1->f__raddRef        = (void (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object**))epv->f__raddRef;
  e1->f__isRemote       = (sidl_bool* (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object**))epv->f__isRemote;
  e1->f__exec           = (void (*)(struct sidl_BaseClass__object*,const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct sidl_BaseInterface__object**))epv->f__exec;
  e1->f_addRef          = (void (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object **))epv->f_addRef;
  e1->f_deleteRef       = (void (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object **))epv->f_deleteRef;
  e1->f_isSame          = (sidl_bool (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object*, struct sidl_BaseInterface__object **))epv->f_isSame;
  e1->f_isType          = (sidl_bool (*)(struct sidl_BaseClass__object*, const char*, struct sidl_BaseInterface__object **))epv->f_isType;
  e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object **))epv->f_getClassInfo;



  s_method_initialized = 1;
  ior_pgas_blockedDoubleArray__ensure_load_called();
}

/**
* pgas_blockedDoubleArray__getEPVs: Get my version of all relevant EPVs.
*/
void pgas_blockedDoubleArray__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
  struct pgas_blockedDoubleArray__epv **s_arg_epv__pgas_blockeddoublearray,
  struct pgas_blockedDoubleArray__epv **s_arg_epv_hooks__pgas_blockeddoublearray
  )
{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    pgas_blockedDoubleArray__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_my_epv__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_my_epv__sidl_baseclass;
  *s_arg_epv__pgas_blockeddoublearray = &s_my_epv__pgas_blockeddoublearray;
  *s_arg_epv_hooks__pgas_blockeddoublearray = &s_my_epv_hooks__pgas_blockeddoublearray;
}

/*
 * __getSuperEPV: returns parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* pgas_blockedDoubleArray__getSuperEPV(void) {
  return s_par_epv__sidl_baseclass;
}

/* no get_static_epv since there are no static methods */

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info, struct sidl_BaseInterface__object **_ex)
{
  LOCK_STATIC_GLOBALS;
  *_ex  = NULL; /* default to no exception */

  if (!s_classInfo) {
    sidl_ClassInfoI impl;
    impl = sidl_ClassInfoI__create(_ex);
    s_classInfo = sidl_ClassInfo__cast(impl,_ex);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "pgas.blockedDoubleArray", _ex);
      sidl_ClassInfoI_setVersion(impl, "1.0", _ex);
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION, _ex);
      sidl_ClassInfoI_deleteRef(impl,_ex);
      sidl_atexit(sidl_deleteRef_atexit, &s_classInfo);
    }
  }
  UNLOCK_STATIC_GLOBALS;
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info,_ex);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info,_ex);
  }
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct pgas_blockedDoubleArray__object* self, sidl_BaseInterface* _ex)
{
  *_ex = 0; /* default no exception */
  if (self) {
    struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data*)((*self).d_sidl_baseclass.d_data);
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo),_ex); SIDL_CHECK(*_ex);
    }
  }
EXIT:
return;
}

/*
 * pgas_blockedDoubleArray__createObject: Allocate the object and initialize it.
 */

struct pgas_blockedDoubleArray__object*
pgas_blockedDoubleArray__createObject(void* ddata, struct sidl_BaseInterface__object ** _ex)
{
  struct pgas_blockedDoubleArray__object* self =
    (struct pgas_blockedDoubleArray__object*) sidl_malloc(
      sizeof(struct pgas_blockedDoubleArray__object),
      "Object allocation failed for struct pgas_blockedDoubleArray__object",
        __FILE__, __LINE__, "pgas_blockedDoubleArray__createObject", _ex);
  if (!self) goto EXIT;
  pgas_blockedDoubleArray__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;

  EXIT:
  return NULL;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void pgas_blockedDoubleArray__init(
  struct pgas_blockedDoubleArray__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{
  struct pgas_blockedDoubleArray__object* s0 = self;
  struct sidl_BaseClass__object*          s1 = &s0->d_sidl_baseclass;

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    pgas_blockedDoubleArray__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  sidl_BaseClass__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

  s1->d_sidl_baseinterface.d_epv = &s_my_epv__sidl_baseinterface;
  s1->d_epv = &s_my_epv__sidl_baseclass;

  s0->d_epv = &s_my_epv__pgas_blockeddoublearray;


  s0->d_epv    = &s_my_epv__pgas_blockeddoublearray;

  s0->d_data = NULL;

  if (ddata) {
    self->d_data = ddata;
    (*(self->d_epv->f__ctor2))(self,ddata,_ex); SIDL_CHECK(*_ex);
  } else { 
    (*(self->d_epv->f__ctor))(self,_ex); SIDL_CHECK(*_ex);
  }
  EXIT:
  return;
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void pgas_blockedDoubleArray__fini(
  struct pgas_blockedDoubleArray__object* self,
  struct sidl_BaseInterface__object **_ex)
{
  struct pgas_blockedDoubleArray__object* s0 = self;
  struct sidl_BaseClass__object*          s1 = &s0->d_sidl_baseclass;

  *_ex  = NULL; /* default to no exception */

  (*(s0->d_epv->f__dtor))(s0,_ex); SIDL_CHECK(*_ex);

  s1->d_sidl_baseinterface.d_epv = s_par_epv__sidl_baseinterface;
  s1->d_epv = s_par_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1, _ex); SIDL_CHECK(*_ex);

  EXIT:
  return;
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
pgas_blockedDoubleArray__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct pgas_blockedDoubleArray__external
s_externalEntryPoints = {
  pgas_blockedDoubleArray__createObject,
  /* no SEPV */
  pgas_blockedDoubleArray__getSuperEPV,
  2, 
  0.0
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct pgas_blockedDoubleArray__external*
pgas_blockedDoubleArray__externals(void)
{
  return &s_externalEntryPoints;
}

