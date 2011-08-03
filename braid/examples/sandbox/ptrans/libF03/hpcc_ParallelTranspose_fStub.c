/*
 * File:          hpcc_ParallelTranspose_fStub.c
 * Symbol:        hpcc.ParallelTranspose-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 7138  trunk)
 * Description:   Client-side glue code for hpcc.ParallelTranspose
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

/*
 * Symbol "hpcc.ParallelTranspose" (version 0.1)
 */

#ifndef included_hpcc_ParallelTranspose_fStub_h
#include "hpcc_ParallelTranspose_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
#ifndef included_sidl_String_h
#include "sidl_String.h"
#endif
#ifndef included_sidl_CastException_h
#include "sidl_CastException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#include <stdio.h>
#ifndef included_sidlf90array_h
#include "sidlf90array.h"
#endif
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_Exception_h
#include "sidl_Exception.h"
#endif
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "hpcc_ParallelTranspose_IOR.h"
#include "hpcc_ParallelTranspose_fAbbrev.h"
#include "hplsupport_BlockCyclicDistArray2dDouble_IOR.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidlOps_h
#include "sidlOps.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_hpcc_ParallelTranspose_fStub_h
#include "hpcc_ParallelTranspose_fStub.h"
#endif
#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_fStub_h
#include "hplsupport_BlockCyclicDistArray2dDouble_fStub.h"
#endif
#ifndef included_sidl_BaseClass_fStub_h
#include "sidl_BaseClass_fStub.h"
#endif
#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
#ifdef WITH_RMI
static struct hpcc_ParallelTranspose__object* 
  hpcc_ParallelTranspose__remoteCreate(const char* url, sidl_BaseInterface 
  *_ex);
static struct hpcc_ParallelTranspose__object* 
  hpcc_ParallelTranspose__remoteConnect(const char* url, sidl_bool ar, 
  sidl_BaseInterface *_ex);
static struct hpcc_ParallelTranspose__object* hpcc_ParallelTranspose__IHConnect(
  struct sidl_rmi_InstanceHandle__object *instance, struct 
  sidl_BaseInterface__object **_ex);
#endif /*WITH_RMI*/
/*
 * Return pointer to internal IOR functions.
 */

static const struct hpcc_ParallelTranspose__external* _getIOR(void)
{
  static const struct hpcc_ParallelTranspose__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = hpcc_ParallelTranspose__externals();
#else
    _ior = (struct hpcc_ParallelTranspose__external*)sidl_dynamicLoadIOR(
      "hpcc.ParallelTranspose","hpcc_ParallelTranspose__externals") ;
    sidl_checkIORVersion("hpcc.ParallelTranspose", _ior->d_ior_major_version, 
      _ior->d_ior_minor_version, 2, 0);
#endif
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct hpcc_ParallelTranspose__sepv* _getSEPV(void)
{
  static const struct hpcc_ParallelTranspose__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Returns a pointer to the internal EPV.
 */

const struct hpcc_ParallelTranspose__epv* hpcc_ParallelTranspose_getEPV(struct 
  hpcc_ParallelTranspose__object *obj) {
  return obj->d_epv;
}

/*
 * Returns a pointer to the internal static EPV.
 */

const struct hpcc_ParallelTranspose__sepv* hpcc_ParallelTranspose_getSEPV() {
  return _getSEPV();
}

/*
 * Returns a reference to the internal data pointer.
 */

void * hpcc_ParallelTranspose_getData(struct hpcc_ParallelTranspose__object 
  *obj) {
  return obj->d_data;
}

/*
 * Stores a reference to the internal data pointer.
 */

void hpcc_ParallelTranspose_setData(struct hpcc_ParallelTranspose__object *obj, 
  void *value) {
  obj->d_data = value;
}

/*
 * Constructor for the class.
 */

void
hpcc_ParallelTranspose_newLocal_c(struct hpcc_ParallelTranspose__object **self, 
  struct sidl_BaseInterface__object ** exception) {
  *self = (_getIOR()->createObject)(NULL,exception);
}



/*
 * Remote Constructor for the class.
 */

void
hpcc_ParallelTranspose_newRemote_c
(
  struct hpcc_ParallelTranspose__object* *self,
  char * url,
  struct sidl_BaseInterface__object* *exception
)
{
#ifdef WITH_RMI
  /* declare proxies */
  /* copy incoming values */
  *self = hpcc_ParallelTranspose__remoteCreate(url, exception);
  /* check exception block */
  /* copy outgoing values */
  /* free resources */
#endif /*WITH_RMI*/
}

/*
 * Data Wrapper for the class.
 */

void
hpcc_ParallelTranspose_wrapObj_m
(
  void ** private_data,
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object *_ior_exception = NULL;
  void* _proxy_private_data = NULL;
  _proxy_private_data = sidl_malloc(SIDL_F90_POINTER_SIZE,
    "Memory allocation failure",
    __FILE__, __LINE__,
    "hpcc_ParallelTranspose_wrapObj_m", &_ior_exception);
  if (_proxy_private_data) { 
    memcpy(_proxy_private_data, private_data, SIDL_F90_POINTER_SIZE);
    *self = (ptrdiff_t) (*(_getIOR()->createObject))(_proxy_private_data,
      &_ior_exception);
  }
  *exception = (ptrdiff_t)_ior_exception;
  if (_ior_exception) *self = 0;
}

/*
 * Remote Connector for the class.
 */

void
hpcc_ParallelTranspose_rConnect_c
(
  struct hpcc_ParallelTranspose__object* *self,
  char * url,
  struct sidl_BaseInterface__object* *exception
)
{
#ifdef WITH_RMI
  /* declare proxies */
  /* copy incoming values */
  *self = hpcc_ParallelTranspose__remoteConnect(url, 1, exception);
  /* check exception block */
  /* copy outgoing values */
  /* free resources */
#endif /*WITH_RMI*/
}
/*
 * Cast method for interface and type conversions.
 */

void
hpcc_ParallelTranspose__cast_c
(
  void * *ref,
  struct hpcc_ParallelTranspose__object* *retval,
  struct sidl_BaseInterface__object* *exception
)
{
#ifdef WITH_RMI
  static int connect_loaded = 0;
#endif /*WITH_RMI*/
  struct sidl_BaseInterface__object  *_base = *((struct 
    sidl_BaseInterface__object **) ref);
  struct sidl_BaseInterface__object *proxy_exception;

  *retval = NULL;

#ifdef WITH_RMI

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("hpcc.ParallelTranspose", (
      void*)hpcc_ParallelTranspose__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

#endif /*WITH_RMI*/

  if (_base) {
    *retval = (
      *_base->d_epv->f__cast)(
      _base->d_object,
      "hpcc.ParallelTranspose", &proxy_exception);
  } else {
    *retval = NULL;
    proxy_exception = NULL;
  }
#ifdef WITH_RMI
  EXIT:
#endif /*WITH_RMI*/
  *exception = proxy_exception;
}

/*
 * Cast method for interface and class type conversions.
 */

void
hpcc_ParallelTranspose__cast2_c
(
  struct hpcc_ParallelTranspose__object* self,
  char * name,
  void * *retval,
  struct sidl_BaseInterface__object* *exception
)
{
  /* declare entry point vector */
  struct hpcc_ParallelTranspose__epv *_epv = NULL;
  /* declare proxies */
  /* copy incoming values */
  _epv = self->d_epv;
  /* method call */
  *retval = 
    (*(_epv->f__cast))(
      self,
      name,
      exception
    );
  /* check exception block */
  /* copy outgoing values */
  /* free resources */
}








/*
 * TRUE if this object is remote, false if local
 */

void
hpcc_ParallelTranspose__isLocal_c
(
  struct hpcc_ParallelTranspose__object* self,
  _Bool *retval,
  struct sidl_BaseInterface__object* *exception
)
{
  /* declare entry point vector */
  struct hpcc_ParallelTranspose__epv *_epv = NULL;
  /* declare proxies */
  sidl_bool _proxy_retval;
  /* copy incoming values */
  _epv = self->d_epv;
  /* method call */
  _proxy_retval = 
    !(*(_epv->f__isRemote))(
      self,
      exception
    );
  /* check exception block */
  /* copy outgoing values */
  *retval = _proxy_retval ? TRUE : FALSE;
  /* free resources */
}



















#ifdef WITH_RMI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif
#ifndef included_sidl_NotImplementedException_h
#include "sidl_NotImplementedException.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t hpcc_ParallelTranspose__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &hpcc_ParallelTranspose__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &hpcc_ParallelTranspose__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &hpcc_ParallelTranspose__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 2;
static const int32_t s_IOR_MINOR_VERSION = 0;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct hpcc_ParallelTranspose__epv s_rem_epv__hpcc_paralleltranspose;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_hpcc_ParallelTranspose__cast(
  struct hpcc_ParallelTranspose__object* self,
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
    cmp = strcmp(name, "hpcc.ParallelTranspose");
    if (!cmp) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct hpcc_ParallelTranspose__object*)self);
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
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*, struct 
      sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*, struct 
        sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct 
      hpcc_ParallelTranspose__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_hpcc_ParallelTranspose__delete(
  struct hpcc_ParallelTranspose__object* self,
  struct sidl_BaseInterface__object* *_ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_hpcc_ParallelTranspose__getURL(
  struct hpcc_ParallelTranspose__object* self, struct 
    sidl_BaseInterface__object* *_ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_hpcc_ParallelTranspose__raddRef(
  struct hpcc_ParallelTranspose__object* self,struct 
    sidl_BaseInterface__object* *_ex)
{
  struct sidl_BaseException__object* netex = NULL;
  /* initialize a new invocation */
  struct sidl_BaseInterface__object* _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    *_ex = (struct sidl_BaseInterface__object*)netex;
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_hpcc_ParallelTranspose__isRemote(
    struct hpcc_ParallelTranspose__object* self, 
    struct sidl_BaseInterface__object* *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_hpcc_ParallelTranspose__set_hooks(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* in */ sidl_bool enable,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hpcc.ParallelTranspose._set_hooks.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* Contract enforcement has not been implemented for remote use. */
/* REMOTE METHOD STUB:_set_contracts */
static void
remote_hpcc_ParallelTranspose__set_contracts(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* in */ sidl_bool enable,
  /* in */ const char* enfFilename,
  /* in */ sidl_bool resetCounters,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_set_contracts", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "enfFilename", enfFilename, 
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "resetCounters", resetCounters, 
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hpcc.ParallelTranspose._set_contracts.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* Contract enforcement has not been implemented for remote use. */
/* REMOTE METHOD STUB:_dump_stats */
static void
remote_hpcc_ParallelTranspose__dump_stats(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* in */ const char* filename,
  /* in */ const char* prefix,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_dump_stats", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex);SIDL_CHECK(
      *_ex);
    sidl_rmi_Invocation_packString( _inv, "prefix", prefix, _ex);SIDL_CHECK(
      *_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hpcc.ParallelTranspose._dump_stats.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_hpcc_ParallelTranspose__exec(
  struct hpcc_ParallelTranspose__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  struct sidl_BaseInterface__object* *_ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_hpcc_ParallelTranspose_addRef(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct hpcc_ParallelTranspose__remote* r_obj = (struct 
      hpcc_ParallelTranspose__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
#ifdef SIDL_DEBUG_REFCOUNT
    fprintf(stderr, "babel: addRef %p new count %d (type %s)\n",
      r_obj, r_obj->d_refcount, 
      "hpcc.ParallelTranspose Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_hpcc_ParallelTranspose_deleteRef(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct hpcc_ParallelTranspose__remote* r_obj = (struct 
      hpcc_ParallelTranspose__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
#ifdef SIDL_DEBUG_REFCOUNT
    fprintf(stderr, "babel: deleteRef %p new count %d (type %s)\n",r_obj, r_obj->d_refcount, "hpcc.ParallelTranspose Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_hpcc_ParallelTranspose_isSame(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hpcc.ParallelTranspose.isSame.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_hpcc_ParallelTranspose_isType(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hpcc.ParallelTranspose.isType.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_hpcc_ParallelTranspose_getClassInfo(
  /* in */ struct hpcc_ParallelTranspose__object*self ,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hpcc_ParallelTranspose__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hpcc.ParallelTranspose.getClassInfo.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str, 
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void hpcc_ParallelTranspose__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct hpcc_ParallelTranspose__epv* epv = &s_rem_epv__hpcc_paralleltranspose;
  struct sidl_BaseClass__epv*         e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*     e1  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast               = remote_hpcc_ParallelTranspose__cast;
  epv->f__delete             = remote_hpcc_ParallelTranspose__delete;
  epv->f__exec               = remote_hpcc_ParallelTranspose__exec;
  epv->f__getURL             = remote_hpcc_ParallelTranspose__getURL;
  epv->f__raddRef            = remote_hpcc_ParallelTranspose__raddRef;
  epv->f__isRemote           = remote_hpcc_ParallelTranspose__isRemote;
  epv->f__set_hooks          = remote_hpcc_ParallelTranspose__set_hooks;
  epv->f__set_contracts      = remote_hpcc_ParallelTranspose__set_contracts;
  epv->f__dump_stats         = remote_hpcc_ParallelTranspose__dump_stats;
  epv->f__ctor               = NULL;
  epv->f__ctor2              = NULL;
  epv->f__dtor               = NULL;
  epv->f_addRef              = remote_hpcc_ParallelTranspose_addRef;
  epv->f_deleteRef           = remote_hpcc_ParallelTranspose_deleteRef;
  epv->f_isSame              = remote_hpcc_ParallelTranspose_isSame;
  epv->f_isType              = remote_hpcc_ParallelTranspose_isType;
  epv->f_getClassInfo        = remote_hpcc_ParallelTranspose_getClassInfo;

  e0->f__cast          = (void* (*)(struct sidl_BaseClass__object*, const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e0->f__delete        = (void (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__delete;
  e0->f__getURL        = (char* (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__getURL;
  e0->f__raddRef       = (void (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__raddRef;
  e0->f__isRemote      = (sidl_bool (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__isRemote;
  e0->f__set_hooks     = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
    struct sidl_BaseInterface__object**)) epv->f__set_hooks;
  e0->f__set_contracts = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
    const char*, sidl_bool, struct sidl_BaseInterface__object**)) 
    epv->f__set_contracts;
  e0->f__dump_stats    = (void (*)(struct sidl_BaseClass__object*, const char*, 
    const char*, struct sidl_BaseInterface__object**)) epv->f__dump_stats;
  e0->f__exec          = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef         = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef      = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame         = (sidl_bool (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e0->f_isType         = (sidl_bool (*)(struct sidl_BaseClass__object*,const 
    char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getClassInfo;

  e1->f__cast          = (void* (*)(void*, const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e1->f__delete        = (void (*)(void*, struct sidl_BaseInterface__object**)) 
    epv->f__delete;
  e1->f__getURL        = (char* (*)(void*, struct 
    sidl_BaseInterface__object**)) epv->f__getURL;
  e1->f__raddRef       = (void (*)(void*, struct sidl_BaseInterface__object**)) 
    epv->f__raddRef;
  e1->f__isRemote      = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object**)) epv->f__isRemote;
  e1->f__set_hooks     = (void (*)(void*, sidl_bool, struct 
    sidl_BaseInterface__object**)) epv->f__set_hooks;
  e1->f__set_contracts = (void (*)(void*, sidl_bool, const char*, sidl_bool, 
    struct sidl_BaseInterface__object**)) epv->f__set_contracts;
  e1->f__dump_stats    = (void (*)(void*, const char*, const char*, struct 
    sidl_BaseInterface__object**)) epv->f__dump_stats;
  e1->f__exec          = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef         = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_addRef;
  e1->f_deleteRef      = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_deleteRef;
  e1->f_isSame         = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e1->f_isType         = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct hpcc_ParallelTranspose__object*
hpcc_ParallelTranspose__remoteConnect(const char *url, sidl_bool ar, struct 
  sidl_BaseInterface__object* *_ex)
{
  struct hpcc_ParallelTranspose__object* self = NULL;

  struct hpcc_ParallelTranspose__object* s0;
  struct sidl_BaseClass__object* s1;

  struct hpcc_ParallelTranspose__remote* r_obj = NULL;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = NULL;
  *_ex = NULL;
  if(url == NULL) {return NULL;}
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    struct hpcc_ParallelTranspose__object* retobj = NULL;
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_BaseInterface bi = (
      sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(objectID,
      _ex); SIDL_CHECK(*_ex);
    (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
    retobj = (struct hpcc_ParallelTranspose__object*) (*bi->d_epv->f__cast)(
      bi->d_object, "hpcc.ParallelTranspose", _ex);
    if(!ar) { 
      (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
    }
    return retobj;
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, 
    "hpcc.ParallelTranspose", ar, _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct hpcc_ParallelTranspose__object*) malloc(
      sizeof(struct hpcc_ParallelTranspose__object));

  r_obj =
    (struct hpcc_ParallelTranspose__remote*) malloc(
      sizeof(struct hpcc_ParallelTranspose__remote));

  if(!self || !r_obj) {
    sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
      _ex);
    SIDL_CHECK(*_ex);
    sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
    sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
      "hpcc.ParallelTranspose.EPVgeneration", _ex);
    SIDL_CHECK(*_ex);
    *_ex = (struct sidl_BaseInterface__object*)ex;
    goto EXIT;
  }

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    hpcc_ParallelTranspose__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__hpcc_paralleltranspose;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(self) { free(self); }
  if(r_obj) { free(r_obj); }
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct hpcc_ParallelTranspose__object*
hpcc_ParallelTranspose__IHConnect(sidl_rmi_InstanceHandle instance, struct 
  sidl_BaseInterface__object* *_ex)
{
  struct hpcc_ParallelTranspose__object* self = NULL;

  struct hpcc_ParallelTranspose__object* s0;
  struct sidl_BaseClass__object* s1;

  struct hpcc_ParallelTranspose__remote* r_obj = NULL;
  self =
    (struct hpcc_ParallelTranspose__object*) malloc(
      sizeof(struct hpcc_ParallelTranspose__object));

  r_obj =
    (struct hpcc_ParallelTranspose__remote*) malloc(
      sizeof(struct hpcc_ParallelTranspose__remote));

  if(!self || !r_obj) {
    sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
      _ex);
    SIDL_CHECK(*_ex);
    sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
    sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
      "hpcc.ParallelTranspose.EPVgeneration", _ex);
    SIDL_CHECK(*_ex);
    *_ex = (struct sidl_BaseInterface__object*)ex;
    goto EXIT;
  }

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    hpcc_ParallelTranspose__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__hpcc_paralleltranspose;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  if(self) { free(self); }
  if(r_obj) { free(r_obj); }
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct hpcc_ParallelTranspose__object*
hpcc_ParallelTranspose__remoteCreate(const char *url, struct 
  sidl_BaseInterface__object **_ex)
{
  struct sidl_BaseInterface__object* _throwaway_exception = NULL;
  struct hpcc_ParallelTranspose__object* self = NULL;

  struct hpcc_ParallelTranspose__object* s0;
  struct sidl_BaseClass__object* s1;

  struct hpcc_ParallelTranspose__remote* r_obj = NULL;
  sidl_rmi_InstanceHandle instance = sidl_rmi_ProtocolFactory_createInstance(
    url, "hpcc.ParallelTranspose", _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct hpcc_ParallelTranspose__object*) malloc(
      sizeof(struct hpcc_ParallelTranspose__object));

  r_obj =
    (struct hpcc_ParallelTranspose__remote*) malloc(
      sizeof(struct hpcc_ParallelTranspose__remote));

  if(!self || !r_obj) {
    sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
      _ex);
    SIDL_CHECK(*_ex);
    sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
    sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
      "hpcc.ParallelTranspose.EPVgeneration", _ex);
    SIDL_CHECK(*_ex);
    *_ex = (struct sidl_BaseInterface__object*)ex;
    goto EXIT;
  }

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    hpcc_ParallelTranspose__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__hpcc_paralleltranspose;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance, 
    &_throwaway_exception); }
  if(self) { free(self); }
  if(r_obj) { free(r_obj); }
  return NULL;
}
/*
 * RMI connector function for the class.
 */

struct hpcc_ParallelTranspose__object*
hpcc_ParallelTranspose__connectI(const char* url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex)
{
  return hpcc_ParallelTranspose__remoteConnect(url, ar, _ex);
}


#endif /*WITH_RMI*/
