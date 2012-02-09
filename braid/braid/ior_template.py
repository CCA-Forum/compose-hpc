## @package ior_template
# template for IOR C code
text = r"""
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
#ifndef included_sidlOps_h
#include "sidlOps.h"
#endif

#include "{Class}_IOR.h"
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
static struct sidl_recursive_mutex_t {Class}__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &{Class}__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &{Class}__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &{Class}__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

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

static struct {Class}__epv s_my_epv__{Class_low} = {{ 0 }};

static struct {Class}__epv s_my_epv_contracts__{Class_low} = {{ 0 }};

static struct {Class}__epv s_my_epv_hooks__{Class_low} = {{ 0 }};

static struct sidl_BaseClass__epv  s_my_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_par_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_my_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_par_epv__sidl_baseinterface;

static struct {Class}__pre_epv s_preEPV;
static struct {Class}__post_epv s_postEPV;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {{
#endif

extern void {Class}__set_epv(
  struct {Class}__epv* epv,
    struct {Class}__pre_epv* pre_epv,
    struct {Class}__post_epv* post_epv);

extern void {Class}__call_load(void);
#ifdef __cplusplus
}}
#endif


/*
 * CHECKS: Enable/disable contract enforcement.
 */

static void ior_{Class}__set_contracts(
  struct {Class}__object* self,
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex)
{{
  *_ex  = NULL;
  {{
    /*
     * Nothing to do since contract enforcement not needed.
     */

  }}
}}

/*
 * DUMP: Dump interface contract enforcement statistics.
 */

static void ior_{Class}__dump_stats(
  struct {Class}__object* self,
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex)
{{
  *_ex = NULL;
  {{
    /*
     * Nothing to do since contract checks not generated.
     */

  }}
}}

static void ior_{Class}__ensure_load_called(void) {{
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {{
    s_load_called=1;
    {Class}__call_load();
  }}
}}

/* CAST: dynamic type casting support. */
static void* ior_{Class}__cast(
  struct {Class}__object* self,
  const char* name, sidl_BaseInterface* _ex)
{{
  int cmp;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp = strcmp(name, "sidl.BaseClass");
  if (!cmp) {{
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = ((struct sidl_BaseClass__object*)self);
    return cast;
  }}
  else if (cmp < 0) {{
    cmp = strcmp(name, "Args.Basic");
    if (!cmp) {{
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct {Class}__object*)self);
      return cast;
    }}
  }}
  else if (cmp > 0) {{
    cmp = strcmp(name, "sidl.BaseInterface");
    if (!cmp) {{
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }}
  }}
  return cast;
  EXIT:
  return NULL;
}}

/*
 * HOOKS: Enable/disable hooks.
 */

static void ior_{Class}__set_hooks(
  struct {Class}__object* self,
  sidl_bool enable, struct sidl_BaseInterface__object **_ex )
{{
  *_ex  = NULL;
  /*
   * Nothing else to do since hook methods not generated.
   */

}}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_{Class}__delete(
  struct {Class}__object* self, struct sidl_BaseInterface__object **_ex)
{{
  *_ex  = NULL; /* default to no exception */
  {Class}__fini(self, _ex);
  memset((void*)self, 0, sizeof(struct {Class}__object));
  free((void*) self);
}}

static char*
ior_{Class}__getURL(
  struct {Class}__object* self,
  struct sidl_BaseInterface__object **_ex)
{{
  char* ret  = NULL;
  char* objid = sidl_rmi_InstanceRegistry_getInstanceByClass((sidl_BaseClass)self, _ex); SIDL_CHECK(*_ex);
  if (!objid) {{
    objid = sidl_rmi_InstanceRegistry_registerInstance((sidl_BaseClass)self, _ex); SIDL_CHECK(*_ex);
  }}
#ifdef WITH_RMI

  ret = sidl_rmi_ServerRegistry_getServerURL(objid, _ex); SIDL_CHECK(*_ex);

#else

  ret = objid;

#endif /*WITH_RMI*/
  return ret;
  EXIT:
  return NULL;
}}
static void
ior_{Class}__raddRef(
    struct {Class}__object* self, sidl_BaseInterface* _ex) {{
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}}

static sidl_bool
ior_{Class}__isRemote(
    struct {Class}__object* self, sidl_BaseInterface* _ex) {{
  *_ex  = NULL; /* default to no exception */
  return FALSE;
}}

struct {Class}__method {{
  const char *d_name;
  void (*d_func)(struct {Class}__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
}};

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void {Class}__init_epv(void)
{{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct {Class}__epv*         epv  = &s_my_epv__{Class_low};
  struct sidl_BaseClass__epv*     e0   = &s_my_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv* e1   = &s_my_epv__sidl_baseinterface;

  struct sidl_BaseClass__epv* s1 = NULL;

  /*
   * Get my parent's EPVs so I can start with their functions.
   */

  sidl_BaseClass__getEPVs(
    &s_par_epv__sidl_baseinterface,
    &s_par_epv__sidl_baseclass);


  /*
   * Alias the static epvs to some handy small names.
   */

  s1  =  s_par_epv__sidl_baseclass;

  epv->f__cast                  = ior_{Class}__cast;
  epv->f__delete                = ior_{Class}__delete;
  epv->f__exec                  = NULL; //ior_{Class}__exec;
  epv->f__getURL                = ior_{Class}__getURL;
  epv->f__raddRef               = ior_{Class}__raddRef;
  epv->f__isRemote              = ior_{Class}__isRemote;
  epv->f__set_hooks             = ior_{Class}__set_hooks;
  epv->f__set_contracts         = ior_{Class}__set_contracts;
  epv->f__dump_stats            = ior_{Class}__dump_stats;
  epv->f_addRef                 = (void (*)(struct {Class}__object*,struct sidl_BaseInterface__object **)) s1->f_addRef;
  epv->f_deleteRef              = (void (*)(struct {Class}__object*,struct sidl_BaseInterface__object **)) s1->f_deleteRef;
  epv->f_isSame                 = (sidl_bool (*)(struct {Class}__object*,struct sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) s1->f_isSame;
  epv->f_isType                 = (sidl_bool (*)(struct {Class}__object*,const char*,struct sidl_BaseInterface__object **)) s1->f_isType;
  epv->f_getClassInfo           = (struct sidl_ClassInfo__object* (*)(struct {Class}__object*,struct sidl_BaseInterface__object **)) s1->f_getClassInfo;

  {Class}__set_epv(epv, &s_preEPV, &s_postEPV);

  /*
   * Override function pointers for sidl.BaseClass with mine, as needed.
   */

  e0->f__cast                 = (void* (*)(struct sidl_BaseClass__object*,const char*, struct sidl_BaseInterface__object**))epv->f__cast;
  e0->f__delete               = (void (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object **)) epv->f__delete;
  e0->f__getURL               = (char* (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object **)) epv->f__getURL;
  e0->f__raddRef              = (void (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e0->f__isRemote             = (sidl_bool (*)(struct sidl_BaseClass__object*, struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e0->f__exec                 = (void (*)(struct sidl_BaseClass__object*,const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef                = (void (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef             = (void (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame                = (sidl_bool (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType                = (sidl_bool (*)(struct sidl_BaseClass__object*,const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo          = (struct sidl_ClassInfo__object* (*)(struct sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) epv->f_getClassInfo;



  /*
   * Override function pointers for sidl.BaseInterface with mine, as needed.
   */

  e1->f__cast                 = (void* (*)(void*,const char*, struct sidl_BaseInterface__object**))epv->f__cast;
  e1->f__delete               = (void (*)(void*, struct sidl_BaseInterface__object **)) epv->f__delete;
  e1->f__getURL               = (char* (*)(void*, struct sidl_BaseInterface__object **)) epv->f__getURL;
  e1->f__raddRef              = (void (*)(void*, struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e1->f__isRemote             = (sidl_bool (*)(void*, struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e1->f__exec                 = (void (*)(void*,const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef                = (void (*)(void*,struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef             = (void (*)(void*,struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame                = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType                = (sidl_bool (*)(void*,const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo          = (struct sidl_ClassInfo__object* (*)(void*,struct sidl_BaseInterface__object **)) epv->f_getClassInfo;



  s_method_initialized = 1;
  ior_{Class}__ensure_load_called();
}}

/*
 * {Class}__getEPVs: Get my version of all relevant EPVs.
 */

void {Class}__getEPVs (
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
  struct {Class}__epv **s_arg_epv__{Class_low},
  struct {Class}__epv **s_arg_epv_hooks__{Class_low})
{{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {{
    {Class}__init_epv();
  }}
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_my_epv__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_my_epv__sidl_baseclass;
  *s_arg_epv__{Class_low} = &s_my_epv__{Class_low};
  *s_arg_epv_hooks__{Class_low} = &s_my_epv_hooks__{Class_low};
}}
/*
 * __getSuperEPV: returns parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* {Class}__getSuperEPV(void) {{
  return s_par_epv__sidl_baseclass;
}}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info, struct sidl_BaseInterface__object **_ex)
{{
  LOCK_STATIC_GLOBALS;
  *_ex  = NULL; /* default to no exception */

  if (!s_classInfo) {{
    sidl_ClassInfoI impl;
    impl = sidl_ClassInfoI__create(_ex);
    s_classInfo = sidl_ClassInfo__cast(impl,_ex);
    if (impl) {{
      sidl_ClassInfoI_setName(impl, "Args.Basic", _ex);
      sidl_ClassInfoI_setVersion(impl, "1.0", _ex);
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION, _ex);
      sidl_ClassInfoI_deleteRef(impl,_ex);
      sidl_atexit(sidl_deleteRef_atexit, &s_classInfo);
    }}
  }}
  UNLOCK_STATIC_GLOBALS;
  if (s_classInfo) {{
    if (*info) {{
      sidl_ClassInfo_deleteRef(*info,_ex);
    }}
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info,_ex);
  }}
}}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct {Class}__object* self, sidl_BaseInterface* _ex)
{{
  *_ex = 0; /* default no exception */
  if (self) {{
    struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data*)((*self).d_sidl_baseclass.d_data);
    if (data) {{
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo),_ex); SIDL_CHECK(*_ex);
    }}
  }}
EXIT:
return;
}}

/*
 * {Class}__createObject: Allocate the object and initialize it.
 */

struct {Class}__object*
{Class}__createObject(void* ddata, struct sidl_BaseInterface__object ** _ex)
{{
  struct {Class}__object* self =
    (struct {Class}__object*) sidl_malloc(
      sizeof(struct {Class}__object),
      "Object allocation failed for struct {Class}__object",
        __FILE__, __LINE__, "{Class}__createObject", _ex);
  if (!self) goto EXIT;
  {Class}__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;

  EXIT:
  return NULL;
}}

/*
 * INIT: initialize a new instance of the class object.
 */

void {Class}__init(
  struct {Class}__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{{
  struct {Class}__object*     s0 = self;
  struct sidl_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {{
    {Class}__init_epv();
  }}
  UNLOCK_STATIC_GLOBALS;

  sidl_BaseClass__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

  s1->d_sidl_baseinterface.d_epv = &s_my_epv__sidl_baseinterface;
  s1->d_epv                      = &s_my_epv__sidl_baseclass;

  s0->d_epv    = &s_my_epv__{Class_low};

  s0->d_data = NULL;

  if (ddata) {{
    self->d_data = ddata;
    (*(self->d_epv->f__ctor2))(self,ddata,_ex); SIDL_CHECK(*_ex);
  }} else {{ 
    (*(self->d_epv->f__ctor))(self,_ex); SIDL_CHECK(*_ex);
  }}
  EXIT:
  return;
}}

/*
 * FINI: deallocate a class instance (destructor).
 */

void {Class}__fini(
  struct {Class}__object* self,
  struct sidl_BaseInterface__object **_ex)
{{
  struct {Class}__object*     s0 = self;
  struct sidl_BaseClass__object* s1 = &s0->d_sidl_baseclass;

  *_ex  = NULL; /* default to no exception */

  (*(s0->d_epv->f__dtor))(s0,_ex); SIDL_CHECK(*_ex);

  s1->d_sidl_baseinterface.d_epv = s_par_epv__sidl_baseinterface;
  s1->d_epv                      = s_par_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1, _ex); SIDL_CHECK(*_ex);

  EXIT:
  return;
}}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
{Class}__IOR_version(int32_t *major, int32_t *minor)
{{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}}

static const struct {Class}__external
s_externalEntryPoints = {{
  {Class}__createObject,
  {Class}__getSuperEPV,
  2, 
  0
}};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct {Class}__external*
{Class}__externals(void)
{{
  return &s_externalEntryPoints;
}}
"""
