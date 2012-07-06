// 
// File:          pgas_blockedDouble3dArray.cxx
// Symbol:        pgas.blockedDouble3dArray-v1.0
// Symbol Type:   class
// Babel Version: 2.0.0 (Revision: 7481 trunk)
// Description:   Client-side glue code for pgas.blockedDouble3dArray
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_pgas_blockedDouble3dArray_hxx
#include "pgas_blockedDouble3dArray.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseException_hxx
#include "sidl_BaseException.hxx"
#endif
#ifndef included_sidl_LangSpecificException_hxx
#include "sidl_LangSpecificException.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
#ifndef included_sidl_CastException_hxx
#include "sidl_CastException.hxx"
#endif
#ifndef included_sidl_rmi_Call_hxx
#include "sidl_rmi_Call.hxx"
#endif
#ifndef included_sidl_rmi_Return_hxx
#include "sidl_rmi_Return.hxx"
#endif
#ifndef included_sidl_rmi_Ticket_hxx
#include "sidl_rmi_Ticket.hxx"
#endif
#ifndef included_sidl_rmi_InstanceHandle_hxx
#include "sidl_rmi_InstanceHandle.hxx"
#endif
#include "sidl_rmi_ConnectRegistry.h"
#include "sidl_String.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.hxx"
#include "sidl_DLL.hxx"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif

#define LANG_SPECIFIC_INIT()
extern "C" {
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
static struct sidl_recursive_mutex_t pgas_blockedDouble3dArray__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &pgas_blockedDouble3dArray__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &pgas_blockedDouble3dArray__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &pgas_blockedDouble3dArray__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

  // Static variables to hold version of IOR
  static const int32_t s_IOR_MAJOR_VERSION = 2;
  static const int32_t s_IOR_MINOR_VERSION = 0;

  // Static variables for managing EPV initialization.
  static int s_remote_initialized = 0;

  static struct pgas_blockedDouble3dArray__epv 
    s_rem_epv__pgas_blockeddouble3darray;

  static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

  static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_pgas_blockedDouble3dArray__cast(
    struct pgas_blockedDouble3dArray__object* self,
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
      cmp = strcmp(name, "pgas.blockedDouble3dArray");
      if (!cmp) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct pgas_blockedDouble3dArray__object*)self);
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
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_pgas_blockedDouble3dArray__delete(
    struct pgas_blockedDouble3dArray__object* self,
    struct sidl_BaseInterface__object* *_ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_pgas_blockedDouble3dArray__getURL(
    struct pgas_blockedDouble3dArray__object* self, struct 
      sidl_BaseInterface__object* *_ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_pgas_blockedDouble3dArray__raddRef(
    struct pgas_blockedDouble3dArray__object* self,struct 
      sidl_BaseInterface__object* *_ex)
  {
    struct sidl_BaseException__object* netex = NULL;
    // initialize a new invocation
    struct sidl_BaseInterface__object* _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
    sidl_rmi_Response _rsvp = NULL;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "addRef", _ex ); SIDL_CHECK(*_ex);
    // send actual RMI request
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
    // Check for exceptions
    netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
    if(netex != NULL) {
      *_ex = (struct sidl_BaseInterface__object*)netex;
      return;
    }

    // cleanup and return
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
    return;
  }

  // REMOTE ISREMOTE: returns true if this object is Remote (it is).
  static sidl_bool
  remote_pgas_blockedDouble3dArray__isRemote(
      struct pgas_blockedDouble3dArray__object* self, 
      struct sidl_BaseInterface__object* *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_pgas_blockedDouble3dArray__set_hooks(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray._set_hooks.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // Contract enforcement has not been implemented for remote use.
  // REMOTE METHOD STUB:_set_contracts
  static void
  remote_pgas_blockedDouble3dArray__set_contracts(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_contracts", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packString( _inv, "enfFilename", enfFilename, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "resetCounters", resetCounters, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray._set_contracts.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // Contract enforcement has not been implemented for remote use.
  // REMOTE METHOD STUB:_dump_stats
  static void
  remote_pgas_blockedDouble3dArray__dump_stats(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_dump_stats", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "filename", filename, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "prefix", prefix, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray._dump_stats.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE EXEC: call the exec function for the object.
  static void remote_pgas_blockedDouble3dArray__exec(
    struct pgas_blockedDouble3dArray__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    struct sidl_BaseInterface__object* *_ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:allocate
  static void
  remote_pgas_blockedDouble3dArray_allocate(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ int32_t size,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "allocate", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "size", size, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray.allocate.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:get
  static double
  remote_pgas_blockedDouble3dArray_get(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* in */ int32_t idx3,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      double _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "get", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "idx1", idx1, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "idx2", idx2, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "idx3", idx3, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray.get.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackDouble( _rsvp, "_retval", &_retval, 
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:set
  static void
  remote_pgas_blockedDouble3dArray_set(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ int32_t idx1,
    /* in */ int32_t idx2,
    /* in */ int32_t idx3,
    /* in */ double val,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "set", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "idx1", idx1, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "idx2", idx2, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "idx3", idx3, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDouble( _inv, "val", val, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray.set.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_pgas_blockedDouble3dArray_addRef(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct pgas_blockedDouble3dArray__remote* r_obj = (struct 
        pgas_blockedDouble3dArray__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
#ifdef SIDL_DEBUG_REFCOUNT
      fprintf(stderr, "babel: addRef %p new count %d (type %s)\n",
        r_obj, r_obj->d_refcount, 
        "pgas.blockedDouble3dArray Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_pgas_blockedDouble3dArray_deleteRef(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct pgas_blockedDouble3dArray__remote* r_obj = (struct 
        pgas_blockedDouble3dArray__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount--;
#ifdef SIDL_DEBUG_REFCOUNT
      fprintf(stderr, "babel: deleteRef %p new count %d (type %s)\n",r_obj, r_obj->d_refcount, "pgas.blockedDouble3dArray Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
      if(r_obj->d_refcount == 0) {
        sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
        free(r_obj);
        free(self);
      }
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:isSame
  static sidl_bool
  remote_pgas_blockedDouble3dArray_isSame(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isSame", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(iobj){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(
          *_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(
          *_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray.isSame.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:isType
  static sidl_bool
  remote_pgas_blockedDouble3dArray_isType(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray.isType.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:getClassInfo
  static struct sidl_ClassInfo__object*
  remote_pgas_blockedDouble3dArray_getClassInfo(
    /* in */ struct pgas_blockedDouble3dArray__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char*_retval_str = NULL;
      struct sidl_ClassInfo__object* _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pgas_blockedDouble3dArray__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pgas.blockedDouble3dArray.getClassInfo.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str, 
        _ex);SIDL_CHECK(*_ex);
      _retval = sidl_ClassInfo__connectI(_retval_str, FALSE, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void pgas_blockedDouble3dArray__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct pgas_blockedDouble3dArray__epv* epv = 
      &s_rem_epv__pgas_blockeddouble3darray;
    struct sidl_BaseClass__epv*            e0  = &s_rem_epv__sidl_baseclass;
    struct sidl_BaseInterface__epv*        e1  = &s_rem_epv__sidl_baseinterface;

    epv->f__cast               = remote_pgas_blockedDouble3dArray__cast;
    epv->f__delete             = remote_pgas_blockedDouble3dArray__delete;
    epv->f__exec               = remote_pgas_blockedDouble3dArray__exec;
    epv->f__getURL             = remote_pgas_blockedDouble3dArray__getURL;
    epv->f__raddRef            = remote_pgas_blockedDouble3dArray__raddRef;
    epv->f__isRemote           = remote_pgas_blockedDouble3dArray__isRemote;
    epv->f__set_hooks          = remote_pgas_blockedDouble3dArray__set_hooks;
    epv->f__set_contracts      = 
      remote_pgas_blockedDouble3dArray__set_contracts;
    epv->f__dump_stats         = remote_pgas_blockedDouble3dArray__dump_stats;
    epv->f__ctor               = NULL;
    epv->f__ctor2              = NULL;
    epv->f__dtor               = NULL;
    epv->f_allocate            = remote_pgas_blockedDouble3dArray_allocate;
    epv->f_get                 = remote_pgas_blockedDouble3dArray_get;
    epv->f_set                 = remote_pgas_blockedDouble3dArray_set;
    epv->f_addRef              = remote_pgas_blockedDouble3dArray_addRef;
    epv->f_deleteRef           = remote_pgas_blockedDouble3dArray_deleteRef;
    epv->f_isSame              = remote_pgas_blockedDouble3dArray_isSame;
    epv->f_isType              = remote_pgas_blockedDouble3dArray_isType;
    epv->f_getClassInfo        = remote_pgas_blockedDouble3dArray_getClassInfo;

    e0->f__cast          = (void* (*)(struct sidl_BaseClass__object*, const 
      char*, struct sidl_BaseInterface__object**)) epv->f__cast;
    e0->f__delete        = (void (*)(struct sidl_BaseClass__object*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e0->f__getURL        = (char* (*)(struct sidl_BaseClass__object*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e0->f__raddRef       = (void (*)(struct sidl_BaseClass__object*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
    e0->f__isRemote      = (sidl_bool (*)(struct sidl_BaseClass__object*, 
      struct sidl_BaseInterface__object**)) epv->f__isRemote;
    e0->f__set_hooks     = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
      struct sidl_BaseInterface__object**)) epv->f__set_hooks;
    e0->f__set_contracts = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
      const char*, sidl_bool, struct sidl_BaseInterface__object**)) 
      epv->f__set_contracts;
    e0->f__dump_stats    = (void (*)(struct sidl_BaseClass__object*, const 
      char*, const char*, struct sidl_BaseInterface__object**)) 
      epv->f__dump_stats;
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
    e1->f__delete        = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e1->f__getURL        = (char* (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e1->f__raddRef       = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
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
    e1->f_addRef         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e1->f_deleteRef      = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e1->f_isSame         = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e1->f_isType         = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct pgas_blockedDouble3dArray__object*
  pgas_blockedDouble3dArray__remoteConnect(const char *url, sidl_bool ar, 
    struct sidl_BaseInterface__object* *_ex)
  {
    struct pgas_blockedDouble3dArray__object* self = NULL;

    struct pgas_blockedDouble3dArray__object* s0;
    struct sidl_BaseClass__object* s1;

    struct pgas_blockedDouble3dArray__remote* r_obj = NULL;
    sidl_rmi_InstanceHandle instance = NULL;
    char* objectID = NULL;
    objectID = NULL;
    *_ex = NULL;
    if(url == NULL) {return NULL;}
    objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
    if(objectID) {
      struct pgas_blockedDouble3dArray__object* retobj = NULL;
      struct sidl_BaseInterface__object *throwaway_exception;
      sidl_BaseInterface bi = (
        sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
        objectID, _ex); SIDL_CHECK(*_ex);
      (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
      retobj = (struct pgas_blockedDouble3dArray__object*) (
        *bi->d_epv->f__cast)(bi->d_object, "pgas.blockedDouble3dArray", _ex);
      if(!ar) { 
        (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
      }
      return retobj;
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, 
      "pgas.blockedDouble3dArray", ar, _ex ); SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct pgas_blockedDouble3dArray__object*) malloc(
        sizeof(struct pgas_blockedDouble3dArray__object));

    r_obj =
      (struct pgas_blockedDouble3dArray__remote*) malloc(
        sizeof(struct pgas_blockedDouble3dArray__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pgas.blockedDouble3dArray.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                     self;
    s1 =                                     &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pgas_blockedDouble3dArray__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pgas_blockeddouble3darray;

    self->d_data = (void*) r_obj;

    return self;
    EXIT:
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // Create an instance that uses an already existing 
  // InstanceHandle to connect to an existing remote object.
  static VAR_UNUSED struct pgas_blockedDouble3dArray__object*
  pgas_blockedDouble3dArray__IHConnect(sidl_rmi_InstanceHandle instance, struct 
    sidl_BaseInterface__object* *_ex)
  {
    struct pgas_blockedDouble3dArray__object* self = NULL;

    struct pgas_blockedDouble3dArray__object* s0;
    struct sidl_BaseClass__object* s1;

    struct pgas_blockedDouble3dArray__remote* r_obj = NULL;
    self =
      (struct pgas_blockedDouble3dArray__object*) malloc(
        sizeof(struct pgas_blockedDouble3dArray__object));

    r_obj =
      (struct pgas_blockedDouble3dArray__remote*) malloc(
        sizeof(struct pgas_blockedDouble3dArray__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pgas.blockedDouble3dArray.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                     self;
    s1 =                                     &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pgas_blockedDouble3dArray__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pgas_blockeddouble3darray;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
    return self;
    EXIT:
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // REMOTE: generate remote instance given URL string.
  static struct pgas_blockedDouble3dArray__object*
  pgas_blockedDouble3dArray__remoteCreate(const char *url, struct 
    sidl_BaseInterface__object **_ex)
  {
    struct sidl_BaseInterface__object* _throwaway_exception = NULL;
    struct pgas_blockedDouble3dArray__object* self = NULL;

    struct pgas_blockedDouble3dArray__object* s0;
    struct sidl_BaseClass__object* s1;

    struct pgas_blockedDouble3dArray__remote* r_obj = NULL;
    sidl_rmi_InstanceHandle instance = sidl_rmi_ProtocolFactory_createInstance(
      url, "pgas.blockedDouble3dArray", _ex ); SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct pgas_blockedDouble3dArray__object*) malloc(
        sizeof(struct pgas_blockedDouble3dArray__object));

    r_obj =
      (struct pgas_blockedDouble3dArray__remote*) malloc(
        sizeof(struct pgas_blockedDouble3dArray__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pgas.blockedDouble3dArray.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                     self;
    s1 =                                     &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pgas_blockedDouble3dArray__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pgas_blockeddouble3darray;

    self->d_data = (void*) r_obj;

    return self;
    EXIT:
    if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance, 
      &_throwaway_exception); }
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // 
  // RMI connector function for the class.
  // 
  struct pgas_blockedDouble3dArray__object*
  pgas_blockedDouble3dArray__connectI(const char* url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex)
  {
    return pgas_blockedDouble3dArray__remoteConnect(url, ar, _ex);
  }


#endif /*WITH_RMI*/
}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
pgas::blockedDouble3dArray::throwException0(
  const char* methodName,
  struct sidl_BaseInterface__object *_exception
)
  // throws:
{
  void * _p = 0;
  struct sidl_BaseInterface__object *throwaway_exception;

  if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object, 
    "sidl.RuntimeException", &throwaway_exception)) != 0 ) {
    struct sidl_RuntimeException__object * _realtype = reinterpret_cast< struct 
      sidl_RuntimeException__object*>(_p);
    (*_exception->d_epv->f_deleteRef)(_exception->d_object, 
      &throwaway_exception);
    // Note: alternate constructor does not increment refcount.
    ::sidl::RuntimeException _resolved_exception = ::sidl::RuntimeException( 
      _realtype, false );
    (_resolved_exception._get_ior()->d_epv->f_add) (
      _resolved_exception._get_ior()->d_object , __FILE__, __LINE__, methodName,
      &throwaway_exception);throw _resolved_exception;
  }
  // Any unresolved exception is treated as LangSpecificException
  ::sidl::LangSpecificException _unexpected = 
    ::sidl::LangSpecificException::_create();
  _unexpected.add(__FILE__,__LINE__, "Unknown method");
  _unexpected.setNote("Unexpected exception received by C++ stub.");
  throw _unexpected;
}

pgas::blockedDouble3dArray::sepv_t *pgas::blockedDouble3dArray::_sepv;


//////////////////////////////////////////////////
// 
// User Defined Methods
// 


/**
 * allocate a blocked cubic array of doubles in sizesizesize
 */
void
pgas::blockedDouble3dArray::allocate( /* in */int32_t size )

{

  ior_t* const loc_self = (struct pgas_blockedDouble3dArray__object*) 
    ::pgas::blockedDouble3dArray::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_allocate))(loc_self, /* in */ size, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("allocate", _exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
double
pgas::blockedDouble3dArray::get( /* in */int32_t idx1, /* in */int32_t idx2, /* 
  in */int32_t idx3 )

{
  double _result;
  ior_t* const loc_self = (struct pgas_blockedDouble3dArray__object*) 
    ::pgas::blockedDouble3dArray::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_get))(loc_self, /* in */ idx1, /* in */ idx2, 
    /* in */ idx3, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("get", _exception);
  }
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
void
pgas::blockedDouble3dArray::set( /* in */int32_t idx1, /* in */int32_t idx2, /* 
  in */int32_t idx3, /* in */double val )

{

  ior_t* const loc_self = (struct pgas_blockedDouble3dArray__object*) 
    ::pgas::blockedDouble3dArray::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_set))(loc_self, /* in */ idx1, /* in */ idx2, /* in */ 
    idx3, /* in */ val, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("set", _exception);
  }
  /*unpack results and cleanup*/
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::pgas::blockedDouble3dArray
pgas::blockedDouble3dArray::_create() {
  struct sidl_BaseInterface__object * _exception = NULL;
  ::pgas::blockedDouble3dArray self( (*_get_ext()->createObject)(NULL, 
    &_exception), false );
  if (_exception != NULL) {
    throwException0("::pgas::blockedDouble3dArray"
      "static constructor", _exception);
  }
  return self;
}

// Internal data wrapping method
::pgas::blockedDouble3dArray::ior_t*
pgas::blockedDouble3dArray::_wrapObj(void* private_data) {
  struct sidl_BaseInterface__object *_exception = NULL;
  ::pgas::blockedDouble3dArray::ior_t* returnValue = (*_get_ext(
    )->createObject)(private_data ,&_exception);
  if (_exception) {
    throwException0("::pgas::blockedDouble3dArray._wrap", _exception);
  }
  return returnValue;
}

#ifdef WITH_RMI
// remote constructor
::pgas::blockedDouble3dArray
pgas::blockedDouble3dArray::_create(const std::string& url) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception = NULL;
  ior_self = pgas_blockedDouble3dArray__remoteCreate( url.c_str(), &_exception 
    );
  if (_exception != NULL ) {
    throwException0("::pgas::blockedDouble3dArray remoteCreate", _exception);
  }
  return ::pgas::blockedDouble3dArray( ior_self, false );
}

#endif /* WITH_RMI */
#ifdef WITH_RMI
// remote connector
::pgas::blockedDouble3dArray
pgas::blockedDouble3dArray::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception = NULL;
  ior_self = pgas_blockedDouble3dArray__remoteConnect( url.c_str(), 
    ar?TRUE:FALSE, &_exception );
  if (_exception != NULL ) {
    throwException0("::pgas::blockedDouble3dArray connect",_exception);
  }
  return ::pgas::blockedDouble3dArray( ior_self, false );
}
#endif /* WITH_RMI */

// copy constructor
pgas::blockedDouble3dArray::blockedDouble3dArray ( const 
  ::pgas::blockedDouble3dArray& original ) {
  _set_ior((struct pgas_blockedDouble3dArray__object*) 
    original.::pgas::blockedDouble3dArray::_get_ior());
  if(d_self) {
    addRef();
  }
  d_weak_reference = false;
}

// assignment operator
::pgas::blockedDouble3dArray&
pgas::blockedDouble3dArray::operator=( const ::pgas::blockedDouble3dArray& rhs 
  ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    _set_ior((struct pgas_blockedDouble3dArray__object*) 
      rhs.::pgas::blockedDouble3dArray::_get_ior());
    if(d_self) {
      addRef();
    }
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
pgas::blockedDouble3dArray::blockedDouble3dArray ( 
  ::pgas::blockedDouble3dArray::ior_t* ior ) : 
  StubBase(reinterpret_cast< void*>(ior))
#ifndef SIDL_FASTCALL_DISABLE_CACHING
#endif
 { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
pgas::blockedDouble3dArray::blockedDouble3dArray ( 
  ::pgas::blockedDouble3dArray::ior_t* ior, bool isWeak ) : 
  StubBase(reinterpret_cast< void*>(ior), isWeak)
#ifndef SIDL_FASTCALL_DISABLE_CACHING
#endif
 { }

// This safe IOR cast addresses Roundup issue475
int ::pgas::blockedDouble3dArray::_set_ior_typesafe( struct 
  sidl_BaseInterface__object *obj,
                                         const ::std::type_info &argtype) { 
  if ( obj == NULL || argtype == typeid(*this) ) {
    // optimized case:  _set_ior() is sufficient
    _set_ior( reinterpret_cast<ior_t*>(obj) );
    return 0;
  } else {
    // Attempt to downcast ior pointer to matching stub type
    ior_t* _my_ptr = NULL;
    if ((_my_ptr = _cast( obj )) == NULL ) {
      return 1;
    } else {
      _set_ior(_my_ptr);
      struct sidl_BaseInterface__object* _throwaway=NULL;
      sidl_BaseInterface_deleteRef(obj,&_throwaway);
      return 0;
    }
  }
}

// exec has special argument passing to avoid #include circularities
void ::pgas::blockedDouble3dArray::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::pgas::blockedDouble3dArray::ior_t* const loc_self = _get_ior();
  struct sidl_BaseInterface__object *throwaway_exception;
  (*loc_self->d_epv->f__exec)(loc_self,
                                methodName.c_str(),
                                inArgs._get_ior(),
                                outArgs._get_ior(),
                                &throwaway_exception);
}


/**
 * Get the URL of the Implementation of this object (for RMI)
 */
::std::string
pgas::blockedDouble3dArray::_getURL(  )
// throws:
//   ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (struct pgas_blockedDouble3dArray__object*) 
    ::pgas::blockedDouble3dArray::_get_ior();
  char * _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f__getURL))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_getURL", _exception);
  }
  if (_local_result) {
    _result = _local_result;
    ::sidl_String_free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}


/**
 * Method to enable/disable method hooks invocation.
 */
void
pgas::blockedDouble3dArray::_set_hooks( /* in */bool enable )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pgas_blockedDouble3dArray__object*) 
    ::pgas::blockedDouble3dArray::_get_ior();
  sidl_bool _local_enable = enable;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_hooks))(loc_self, /* in */ _local_enable, 
    &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_set_hooks", _exception);
  }
  /*unpack results and cleanup*/
}


/**
 * Method to enable/disable interface contract enforcement.
 */
void
pgas::blockedDouble3dArray::_set_contracts( /* in */bool enable, /* in */const 
  ::std::string& enfFilename, /* in */bool resetCounters )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pgas_blockedDouble3dArray__object*) 
    ::pgas::blockedDouble3dArray::_get_ior();
  sidl_bool _local_enable = enable;
  sidl_bool _local_resetCounters = resetCounters;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_contracts))(loc_self, /* in */ _local_enable, /* 
    in */ enfFilename.c_str(), /* in */ _local_resetCounters, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_set_contracts", _exception);
  }
  /*unpack results and cleanup*/
}


/**
 * Method to dump contract enforcement statistics.
 */
void
pgas::blockedDouble3dArray::_dump_stats( /* in */const ::std::string& filename, 
  /* in */const ::std::string& prefix )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pgas_blockedDouble3dArray__object*) 
    ::pgas::blockedDouble3dArray::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__dump_stats))(loc_self, /* in */ filename.c_str(), /* 
    in */ prefix.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_dump_stats", _exception);
  }
  /*unpack results and cleanup*/
}

// protected method that implements casting
struct pgas_blockedDouble3dArray__object* pgas::blockedDouble3dArray::_cast(
  const void* src)
{
  ior_t* cast = NULL;
#ifdef WITH_RMI
  static int connect_loaded = 0;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("pgas.blockedDouble3dArray", (
      void*)pgas_blockedDouble3dArray__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
#endif /* WITH_RMI */
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object, 
      "pgas.blockedDouble3dArray", &throwaway_exception));
  }
  return cast;
}

// Static data type
const ::pgas::blockedDouble3dArray::ext_t * pgas::blockedDouble3dArray::s_ext = 
  0;

// private static method to get static data type
const ::pgas::blockedDouble3dArray::ext_t *
pgas::blockedDouble3dArray::_get_ext()
  throw ( ::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = pgas_blockedDouble3dArray__externals();
#else
    s_ext = (struct pgas_blockedDouble3dArray__external*)sidl_dynamicLoadIOR(
      "pgas.blockedDouble3dArray","pgas_blockedDouble3dArray__externals") ;
#endif
    sidl_checkIORVersion("pgas.blockedDouble3dArray", 
      s_ext->d_ior_major_version, s_ext->d_ior_minor_version, 2, 0);
  }
  return s_ext;
}

