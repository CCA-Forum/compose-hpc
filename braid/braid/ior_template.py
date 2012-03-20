#!/usr/bin/env python
# -*- python -*-
## @package ior_template
# template for IOR C code
#
# \authors <pre>
#
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
# 
# LLNL-CODE-473891.
# All rights reserved.
#
# This file is part of BRAID. For details, see
# http://compose-hpc.sourceforge.net/.
# Please read the COPYRIGHT file for Our Notice and
# for the BSD License.
#
# </pre>
#

import config, sidl
from sidl_symbols import (
    get_parent_interfaces, 
    has_parent_interface, 
    get_direct_parent_interfaces, 
    get_parent,
    get_unique_interfaces,
    visit_hierarchy)
from string import Template

def gen_IOR_c(symbol_table, is_abstract, has_static_methods, iorname, sorted_parents, cls):
    """
    generate a Babel-style $classname_IOR.c
    """
    return Template(text).substitute(
        CLASS = iorname, 
        CLASS_LOW = str.lower(iorname),
        Casts = cast_binary_search(
            symbol_table, sorted_parents, cls, True),
        Baseclass = baseclass(symbol_table, cls),
        EPVinits = EPVinits(symbol_table, is_abstract, cls),
        EPVfini = EPVfini(symbol_table, is_abstract, cls),
        ParentDecls = ParentDecls(symbol_table, cls),
        StaticEPVDecls = StaticEPVDecls(symbol_table, sorted_parents, cls, 
                                        has_static_methods, is_abstract, iorname),
        External_getSEPV = ('%s__getStaticEPV,'%iorname) if has_static_methods else '/* no SEPV */',
        HAVE_STATIC = '1' if has_static_methods else '0',
        IOR_MAJOR = config.BABEL_VERSION[0], # this will break at Babel 10.0!
        IOR_MINOR = config.BABEL_VERSION[2:]
        )

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

#include "${CLASS}_IOR.h"
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
static struct sidl_recursive_mutex_t ${CLASS}__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &${CLASS}__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &${CLASS}__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &${CLASS}__mutex )==EDEADLOCK) */
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
static int s_static_initialized = 0;

static struct ${CLASS}__epv s_my_epv__${CLASS_LOW} = { 0 };
static struct ${CLASS}__epv s_my_epv_contracts__${CLASS_LOW} = { 0 };
static struct ${CLASS}__epv s_my_epv_hooks__${CLASS_LOW} = { 0 };

${StaticEPVDecls}

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void ${CLASS}__set_epv(
  struct ${CLASS}__epv* epv,
    struct ${CLASS}__pre_epv* pre_epv,
    struct ${CLASS}__post_epv* post_epv);

extern void ${CLASS}__call_load(void);
#ifdef __cplusplus
}
#endif


/*
 * CHECKS: Enable/disable contract enforcement.
 */

static void ior_${CLASS}__set_contracts(
  struct ${CLASS}__object* self,
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex)
{
  *_ex  = NULL;
  {
    /*
     * Nothing to do since contract enforcement not needed.
     */

  }
}

/*
 * DUMP: Dump interface contract enforcement statistics.
 */

static void ior_${CLASS}__dump_stats(
  struct ${CLASS}__object* self,
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex)
{
  *_ex = NULL;
  {
    /*
     * Nothing to do since contract checks not generated.
     */

  }
}

static void ior_${CLASS}__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    s_load_called=1;
    ${CLASS}__call_load();
  }
}

/* CAST: dynamic type casting support. */
static void* ior_${CLASS}__cast(
  struct ${CLASS}__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int cmp;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  ${Casts}
  return cast;
  EXIT:
  return NULL;
}


/*
 * HOOKS: Enable/disable hooks.
 */

static void ior_${CLASS}__set_hooks(
  struct ${CLASS}__object* self,
  sidl_bool enable, struct sidl_BaseInterface__object **_ex )
{
  *_ex  = NULL;
  /*
   * Nothing else to do since hook methods not generated.
   */

}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_${CLASS}__delete(
  struct ${CLASS}__object* self, struct sidl_BaseInterface__object **_ex)
{
  *_ex  = NULL; /* default to no exception */
  ${CLASS}__fini(self, _ex);
  memset((void*)self, 0, sizeof(struct ${CLASS}__object));
  free((void*) self);
}

static char*
ior_${CLASS}__getURL(
  struct ${CLASS}__object* self,
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
ior_${CLASS}__raddRef(
    struct ${CLASS}__object* self, sidl_BaseInterface* _ex) {
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}

static sidl_bool
ior_${CLASS}__isRemote(
    struct ${CLASS}__object* self, sidl_BaseInterface* _ex) {
  *_ex  = NULL; /* default to no exception */
  return FALSE;
}

struct ${CLASS}__method {
  const char *d_name;
  void (*d_func)(struct ${CLASS}__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
};

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void ${CLASS}__init_epv(void)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct ${CLASS}__epv*         epv  = &s_my_epv__${CLASS_LOW};
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

  epv->f__cast                  = ior_${CLASS}__cast;
  epv->f__delete                = ior_${CLASS}__delete;
  epv->f__exec                  = NULL; //ior_${CLASS}__exec;
  epv->f__getURL                = ior_${CLASS}__getURL;
  epv->f__raddRef               = ior_${CLASS}__raddRef;
  epv->f__isRemote              = ior_${CLASS}__isRemote;
  epv->f__set_hooks             = ior_${CLASS}__set_hooks;
  epv->f__set_contracts         = ior_${CLASS}__set_contracts;
  epv->f__dump_stats            = ior_${CLASS}__dump_stats;
  epv->f_addRef                 = (void (*)(struct ${CLASS}__object*,struct sidl_BaseInterface__object **)) s1->f_addRef;
  epv->f_deleteRef              = (void (*)(struct ${CLASS}__object*,struct sidl_BaseInterface__object **)) s1->f_deleteRef;
  epv->f_isSame                 = (sidl_bool (*)(struct ${CLASS}__object*,struct sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) s1->f_isSame;
  epv->f_isType                 = (sidl_bool (*)(struct ${CLASS}__object*,const char*,struct sidl_BaseInterface__object **)) s1->f_isType;
  epv->f_getClassInfo           = (struct sidl_ClassInfo__object* (*)(struct ${CLASS}__object*,struct sidl_BaseInterface__object **)) s1->f_getClassInfo;

  ${CLASS}__set_epv(epv, &s_preEPV, &s_postEPV);

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
  ior_${CLASS}__ensure_load_called();
}

/*
 * ${CLASS}__getEPVs: Get my version of all relevant EPVs.
 */

void ${CLASS}__getEPVs (
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,
  struct ${CLASS}__epv **s_arg_epv__${CLASS_LOW},
  struct ${CLASS}__epv **s_arg_epv_hooks__${CLASS_LOW})
{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    ${CLASS}__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_my_epv__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_my_epv__sidl_baseclass;
  *s_arg_epv__${CLASS_LOW} = &s_my_epv__${CLASS_LOW};
  *s_arg_epv_hooks__${CLASS_LOW} = &s_my_epv_hooks__${CLASS_LOW};
}
/*
 * __getSuperEPV: returns parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* ${CLASS}__getSuperEPV(void) {
  return s_par_epv__sidl_baseclass;
}

/*
 * ${CLASS}__getStaticEPV: return pointer to static EPV structure.
 */
#if ${HAVE_STATIC}
struct ${CLASS}__sepv*
${CLASS}__getStaticEPV(void){
  struct ${CLASS}__sepv* sepv;
  LOCK_STATIC_GLOBALS;
  if (!s_static_initialized) {
    ${CLASS}__init_sepv();
  }
  UNLOCK_STATIC_GLOBALS;

  if (sidl_Enforcer_areEnforcing()) {
    if (!s_cstats.enabled) {
      struct sidl_BaseInterface__object *tae;
      ior_${CLASS}__set_contracts_static(sidl_Enforcer_areEnforcing(),
        NULL, TRUE, &tae);
    }
    sepv = &s_stc_epv_contracts__vect_utils;
  } else {
    sepv = &s_stc_epv__vect_utils;
  }
  return sepv;
}
#endif

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
      sidl_ClassInfoI_setName(impl, "Args.Basic", _ex);
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
initMetadata(struct ${CLASS}__object* self, sidl_BaseInterface* _ex)
{
  *_ex = 0; /* default no exception */
  if (self) {
    struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data*)((*self)${Baseclass}.d_data);
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
 * ${CLASS}__createObject: Allocate the object and initialize it.
 */

struct ${CLASS}__object*
${CLASS}__createObject(void* ddata, struct sidl_BaseInterface__object ** _ex)
{
  struct ${CLASS}__object* self =
    (struct ${CLASS}__object*) sidl_malloc(
      sizeof(struct ${CLASS}__object),
      "Object allocation failed for struct ${CLASS}__object",
        __FILE__, __LINE__, "${CLASS}__createObject", _ex);
  if (!self) goto EXIT;
  ${CLASS}__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;

  EXIT:
  return NULL;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void ${CLASS}__init(
  struct ${CLASS}__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{
${ParentDecls}

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    ${CLASS}__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  sidl_BaseClass__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

${EPVinits}

  s0->d_epv    = &s_my_epv__${CLASS_LOW};

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

void ${CLASS}__fini(
  struct ${CLASS}__object* self,
  struct sidl_BaseInterface__object **_ex)
{
${ParentDecls}

  *_ex  = NULL; /* default to no exception */

${EPVfini}

  EXIT:
  return;
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
${CLASS}__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct ${CLASS}__external
s_externalEntryPoints = {
  ${CLASS}__createObject,
  ${External_getSEPV}
  ${CLASS}__getSuperEPV,
  ${IOR_MAJOR}, 
  ${IOR_MINOR}
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct ${CLASS}__external*
${CLASS}__externals(void)
{
  return &s_externalEntryPoints;
}
"""

# the following functions were ported from the Java Babel implementation

def cast_binary_search(symbol_table, sorted_types, cls, addref):
    """
    Generate the cast function for a class. This will return null if
    the cast is invalid and a pointer to the object otherwise. The
    logic generates tests for the current class and then recursively
    queries the parent classes.
    """

    def bs(ind, lower, upper, addref):
        r = []
        if lower < upper:
            middle = (lower + upper) / 2;
            e = sorted_types[middle];
            r += [ind+'cmp = strcmp(name, "%s");'%qual_id(e[1]),
                  ind+'if (!cmp) {',
                  ind+'  (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);' if addref else '',
                  ind+'  cast = %s;' % class_to_interface_ptr(symbol_table, cls, e),
                  ind+'  return cast;',
                  ind+'}']
            if (lower < middle):
                r += ([ind+'else if (cmp < 0) {'] 
                      + bs(ind+'  ', lower, middle, addref)
                      + [ind+'}'])
            if (middle+1 < upper):
                r += ([ind+'else if (cmp > 0) {']
                      + bs(ind+'  ', middle+1, upper, addref)
                      + [ind+'}'])

        return r
        
    r = bs('  ', 0, len(sorted_types), addref)
    s = '\n'.join(r)
    return s

def class_to_interface_ptr(symbol_table, cls, e):
    """
    Generate an expression to obtain a pointer to an interface or
    subcls from an object pointer.
   
    @param symbol_table    the class's symbol table
    @param cls             the object pointer this is a class pointer to
                           this type
    @param e               this is the type of the interface/subclass pointer
                           to be obtained
    @return  a String containing the expression to cast & (if necessary)
             dereference the this pointer to the appropriate internal
             data structure
    """

    def hasAncestor(excluded, search, target):
        # endless recursion here
        #print 'search=%r, target=%r' %( search, target)
        hsearch = sidl.hashable_type_id(search)
        if hsearch in excluded: return False
        if hsearch == target: return True
        for e in get_parent_interfaces(symbol_table, search):
            if hasAncestor(excluded, symbol_table[e], target):
                return True
        return False

    def nextAncestor(ancestor, result):
        ancestor = get_parent(symbol_table, ancestor)
        if ancestor:
            result.append(".d_")
            result.append(qual_name_low(symbol_table, ancestor))
        return ancestor

    def directlyImplements(cls, e):
        while cls:
            if sidl.type_id(e) in get_direct_parent_interfaces(symbol_table, cls):
                return True
            cls = get_parent(symbol_table, cls)
        return False

    def implementsByInheritance(cls, e):
        parent = get_parent(symbol_table, cls)
        if parent:
            excludedInterfaces = get_parent_interfaces(symbol_table, parent)
        else:
            excludedInterfaces = []

        for ext in get_unique_interfaces(symbol_table, cls):
            if hasAncestor(excludedInterfaces, symbol_table[ext], sidl.hashable_type_id(e)):
                return True
            
        return False

    #import pdb; pdb.set_trace()
    if (symbol_table.prefix, sidl.type_id(cls), '' == sidl.type_id(e) # names are identical 
        or hasAncestor([], cls, sidl.hashable_type_id(e))): 
      if e[0] == sidl.class_:
          # fixme: for enums, this is not true
          return '((struct %s__object*)self)'%qual_name(symbol_table, e)
      
      else:
        ancestor = cls
        result = []
        direct = directlyImplements(cls, e)
        result.append('&((*self)')
        while ancestor:
            if ((direct and (sidl.type_id(e) in get_unique_interfaces(symbol_table, ancestor))) 
                or ((not direct) and implementsByInheritance(ancestor, e))):
                result.append('.d_')
                result.append(qual_name_low(symbol_table, e))
                break
            else:
                ancestor = nextAncestor(ancestor, result)

        if ancestor == None:
            raise Exception('Illegal symbol table entry: ' 
                            + cls.getFullName() + ' and ' 
                            + e.getFullName())
        
        result.append(')')
        return ''.join(result)
      
    else:
        return 'NULL ##ERROR>??##'
    

def qual_id(scoped_id, sep='.'):
    _, prefix, name, ext = scoped_id
    return sep.join(list(prefix)+[name])+ext  
  
def qual_id_low(scoped_id, sep='.'):
    return str.lower(qual_id(scoped_id, sep))

def qual_name(symbol_table, cls):
    assert not sidl.is_scoped_id(cls)
    _, prefix, name, ext = sidl.get_scoped_id(symbol_table, cls)
    return '_'.join(prefix+[name])+ext

def qual_name_low(symbol_table, cls):
    return str.lower(qual_name(symbol_table, cls))


def baseclass(symbol_table, cls):
    r = []
    parent = get_parent(symbol_table, cls)
    while parent:
        r.append('.d_'+qual_name_low(symbol_table, parent))
        parent = get_parent(symbol_table, parent)
    return ''.join(r)


def ParentDecls(symbol_table, cls):
    """
    Recursively output self pointers to the SIDL objects for this class and its
    parents. The self pointers are of the form sN, where N is an integer
    represented by the level argument. If the width is zero, then the width of
    all parents is generated automatically.
    """

    # Calculate the width of this class and all parents for pretty output.
    # Ooh, very pretty.
    width = 0
    parent = cls
    while parent:
        w = len(qual_name(symbol_table, parent))
        if w > width:
            width = w
            
        parent = get_parent(symbol_table, parent)

    def generateParentSelf(cls, level):
        if cls:
            # Now use the width information to print out symbols.
            typ = qual_name(symbol_table, cls)
            if level == 0:
                r.append('  struct %s__object*'%typ+' '*(width-len(typ))+' s0 = self;')
            else:
                r.append('  struct %s__object*'%typ+' '*(width-len(typ))+' s%d = &s%d->d_%s;' 
                         % (level,level-1,qual_name_low(symbol_table, cls)))
     
            generateParentSelf(get_parent(symbol_table, cls), level + 1)

    r = []
    generateParentSelf(cls, 0)
    return '\n'.join(r)

def StaticEPVDecls(symbol_table, parents, cls, has_static_methods, is_abstract, ior_name):
    """
    Collect all the parents of the class in a set and output EPV structures
    for the parents.
    """
    r = []

    # The class
    t = ior_name
    n = str.lower(t)
    if has_static_methods:
        r.append('static VAR_UNUSED struct %s__sepv  s_stc_epv__%s;' % (t, n))
        if generateContractEPVs(symbol_table, is_abstract, cls):
            r.append('static VAR_UNUSED struct %s__sepv  s_stc_epv_contracts__%s;' % (t, n))

    # Interfaces and parents
    new_interfaces = get_unique_interfaces(symbol_table, cls)
    for parent in parents:
        is_par   = not sidl.hashable_type_id(parent) in new_interfaces
        t = qual_id(sidl.type_id(parent), '_')
        n = str.lower(t)
        r.append('static struct %s__epv  s_my_epv__%s;'% (t, n))
        if generateHookEPVs(symbol_table, parent):
            r.append('static struct %s__epv  s_my_pre_epv_hooks__%s;'% (t, n))
        if is_par:
          r.append('static struct %s__epv*  s_par_epv__%s;'% (t, n))
          if generateHookEPVs(symbol_table, parent):
              r.append('static struct %s_pre__epv*  s_par_epv_hooks__%s;'% (t, n))
        r.append('')

    if has_static_methods and (
        generateContractEPVs(symbol_table, is_abstract, cls) or
        generateHookEPVs(symbol_table, cls)):
        r.append('/* Static variables for interface contract enforcement and/or hooks controls. */')
        r.append('static VAR_UNUSED struct %s__cstats s_cstats;' % ior_name)
    
    if generateContractChecks(symbol_table, is_abstract, cls):
        r.append('/* Static file for interface contract enforcement statistics.')
        r.append('static FILE* s_dump_fptr = NULL;')
        r.append('')
 
    # Declare static hooks epvs
    if generateHookEPVs(symbol_table, cls):
        r.append('static struct %s__pre_epv s_preEPV;'% ior_name)
        r.append('static struct %s__post_epv s_postEPV;'% ior_name)
 
        if has_static_methods:
            r.append('static struct %s__pre_sepv s_preSEPV;'% ior_name)
            r.append('static struct %s__post_sepv s_postSEPV;'% ior_name)

        r.append('')
 
    if False: #fastcall:
        r.append('/* used for initialization of native epv entries. */')
        r.append('static const sidl_babel_native_epv_t NULL_NATIVE_EPV  = { BABEL_LANG_UNDEF, NULL};')

    return '\n'.join(r)

def EPVinits(symbol_table, is_abstract, cls):
    r = []
    fixEPVs(r, symbol_table, cls, is_abstract, 0, is_new=True)
    return '\n'.join(r)
                 
                     
def fixEPVs(r, symbol_table, cls, is_abstract, level, is_new):
    """
    Recursively modify the EPVs in parent classes and set up interface
    pointers. Nothing is done if the class argument is null. The flag is_new
    determines whether the EPVs are being set to a newly defined EPV or to a
    previously saved EPV.
    """
    if not cls: return

    parent = get_parent(symbol_table, cls)
    fixEPVs(r, symbol_table, parent, is_abstract, level + 1, is_new)

    # Update the EPVs for all of the new interfaces in this particular class.
    _self    = 's%d' %level
    epvType  = 'my_' if is_new else 'par_'
    prefix   = '&' if is_new else ''
    ifce    = sorted(get_unique_interfaces(symbol_table, cls))
    epv     = 'epv'
    #width   = Utilities.getWidth(ifce) + epv.length() + 3;

    for i in ifce:
        name        = qual_id_low(i, '_')
        r.append('  %s->d_%s.d_%s = %ss_%s%s__%s;' %(_self, name, epv, prefix, epvType, epv, name))

    name_low = qual_name_low(symbol_table, cls)
    name = qual_name(symbol_table, cls)
    
    # Modify the class entry point vector.
    setContractsEPV =  level == 0 and is_new
    if setContractsEPV:
        r.append('')
        r.append('#ifdef SIDL_CONTRACTS_DEBUG')
        r.append(r'  printf("Setting epv...areEnforcing=%d\n", ')
        r.append('  sidl_Enforcer_areEnforcing());');
        r.append('#endif /* SIDL_CONTRACTS_DEBUG */');
        r.append('  if (sidl_Enforcer_areEnforcing()) {');
        r.append('    if (!self->d_cstats.use_hooks) {');
        r.append('#ifdef SIDL_CONTRACTS_DEBUG')
        r.append(r'      printf("Calling set_contracts()...\n");');
        r.append('#endif /* SIDL_CONTRACTS_DEBUG */');
        r.append('      sidl_BaseInterface* tae;');
        r.append('      ior_%s__set_contracts(%s, sidl_Enforcer_areEnforcing(), NULL, TRUE, &tae);' 
                 % (name, _self))
        r.append('    }');
        #  TBD:  Should the Base EPV also be set to a contracts version here?
        #        Can't remember off-hand.
        r.append('#ifdef SIDL_CONTRACTS_DEBUG')
        r.append(r'    printf("Setting epv to contracts version...\n");');
        r.append('#endif /* SIDL_CONTRACTS_DEBUG */');
        r.append('    %s->d_%s = %ss_%s%s__%s;' %(_self, epv, prefix, epvType, epv, name_low))
        r.append('  } else {');
        r.append('#ifdef SIDL_CONTRACTS_DEBUG')
        r.append(r'   printf("Setting epv to regular version...\n");');
        r.append('#endif /* SIDL_CONTRACTS_DEBUG */');

    ind = '  ' if setContractsEPV else ''
    r.append('%s  %s->d_%s = %ss_%s%s__%s;'%(ind, _self, epv, prefix, epvType, epv, name_low))
    if setContractsEPV:
        r.append('  }')

    if generateBaseEPVAttr(symbol_table, is_abstract, cls):
        r.append('  %s->d_%s = %ss_%s%s__%s;'%(_self, epv, 'b', prefix, epvType, name_low))

    r.append('')
                

def EPVfini(symbol_table, is_abstract, cls):
    r = []
    # Dump statistics (if enforcing contracts).
    if generateContractChecks(symbol_table, is_abstract, cls):
      r.append('  if (sidl_Enforcer_areEnforcing()) {')
      r.append('    (*(s0->d_epv->f__dump_stats))(s0, "", "FINI",_ex); SIDL_CHECK(*_ex);')
      r.append('  }');
      r.append('');

    # Call the user-defined destructor for this class.
    r.append('  (*(s0->d_epv->f__dtor))(s0,_ex); SIDL_CHECK(*_ex);')
 
    # If there is a parent class, then reset all parent pointers and call the
    # parent destructor.
    parent = get_parent(symbol_table, cls)
    if parent:
        r.append('')
        fixEPVs(r, symbol_table, parent, is_abstract, 1, is_new=False)
        r.append('  %s__fini(s1, _ex); SIDL_CHECK(*_ex);'%qual_name(symbol_table, parent) );

    return '\n'.join(r)


def generateHookMethods(symbol_table, ext):
    """
    Return TRUE if hook methods are to be generated; FALSE otherwise.
    
    Assumptions:
    1) Assumptions in generateHookEPVs() apply.
    2) Hook methods are only generated if configuration indicates
       their generation is required.
    """
    return generateHookEPVs(symbol_table, ext) and False #context.getConfig().generateHooks();

   
def generateHookEPVs(symbol_table, ext):
   """
   Return TRUE if the hooks-related EPVs are supposed to be generated.  
   
   Assumption:  Only non-SIDL interfaces and classes are to include 
   the hook EPVs.  Exceptions are _not_ to be included.
   """
   s_id = sidl.get_scoped_id(symbol_table, ext)
   return (not isSIDLSymbol(s_id)
           and not isSIDLXSymbol(s_id) 
           and not isException(symbol_table, ext))

def generateBaseEPVAttr(symbol_table, is_abstract, ext):
    """
    Return TRUE if the base EPV attribute needs to be supported; FALSE 
    otherwise.
    """   
    return generateHookMethods(symbol_table, ext) and \
        generateContractChecks(symbol_table, is_abstract, ext)
   

def generateContractChecks(symbol_table, is_abstract, ext):
    """
    Return TRUE if contract checks are supposed to be generated.
    
    Assumptions:
    1) Assumptions in generateContractEPVs() apply.
    2) Checks are only generated if the class has its own or
       inherited contract clauses.
    3) Checks are only generated if the configuration indicates
       their generation is required.
    """     
    return generateContractEPVs(symbol_table, is_abstract, ext) \
        and class_contracts(symbol_table, ext) and \
        True #and context.getConfig().generateContracts();
   

def class_contracts(symbol_table, cls):
    """
    Return TRUE if the class has any invariants or any methods define
    contracts
    """
    has_contracts = False

    def evaluate(sid):
        ext = symbol_table[sid]
        if sidl.ext_invariants(ext):
            has_contracts = True
            return

        for m in sidl.ext_methods(ext):
            if sidl.method_requires(m) or sidl.method_ensures(m):
                has_contracts = True
                return

    if not has_contracts:
        visit_hierarchy(cls, evaluate, symbol_table, [])

    return has_contracts

def generateContractEPVs(symbol_table, is_abstract, ext):
    """
    Return TRUE if the contract-related EPVs are supposed to be generated.  

    Assumptions:  
    1) Contract-related EPVs are only generated for concrete classes.
    2) Contract-related EPVs are not generated for SIDL classes.
    3) Contract-related EPVs are not generated for exceptions.
    """   
    return (not is_abstract) and (not sidl.is_interface(ext)) \
        and generateContractBuiltins(symbol_table, ext)

def isSIDLSymbol(scoped_id):
    """
    Return TRUE if the Symbol ID corresponds to a SIDL symbol; FALSE 
    otherwise.
    """
    return sidl.scoped_id_modules(scoped_id)[0] == 'sidl'
   

def isSIDLXSymbol(scoped_id):
    """
    Return TRUE if the Symbol ID corresponds to a SIDLX symbol; FALSE 
    otherwise.
    """
    return sidl.scoped_id_modules(scoped_id)[0] == 'sidlx'

def generateContractBuiltins(symbol_table, ext):
    """
    Return TRUE if the contract-related built-in methods are to be
    generated.
    Assumptions:  
    1) Contract-related EPVs are not generated for SIDL interfaces or classes.
    2) Contract-related EPVs are not generated for exceptions.
    """
    s_id = sidl.get_scoped_id(symbol_table, ext)
    return ((not isSIDLSymbol(s_id))
            and (not isSIDLXSymbol(s_id))
            and (not isException(symbol_table, ext)))
   
   

def isException(symbol_table, ext):
   """
   Return <code>true</code> if and only if the extendable is
   a class that is the base exception class, is an interface that is
   the base exception interface, or it has the base exception class or 
   interface in its type ancestry.
   """
   base_ex = sidl.Scoped_id(['sidl'], 'BaseException', '')
   sid = sidl.type_id(ext)
   return sid == base_ex or \
       has_parent_interface(symbol_table, ext, base_ex)
