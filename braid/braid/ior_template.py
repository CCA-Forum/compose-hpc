#!/usr/bin/env python
# -*- python -*-
## @package ior_template
# template for IOR C code
#
# Many of the functions in here were directly ported from the Java
# Babel implementation.
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

import config, sidl, ir, babel
from codegen import sidl_gen, c_gen
from patmat import *
from sidl_symbols import visit_hierarchy
from sidlobjects import make_extendable
from string import Template
from utils import accepts

gen_hooks = False
gen_contracts = True
skip_rmi = False

def gen_IOR_c(iorname, cls, _gen_hooks, _gen_contracts):
    """
    generate a Babel-style $classname_IOR.c
    """
    global gen_hooks
    gen_hooks = _gen_hooks
    global gen_contracts
    gen_contracts = _gen_contracts
    # class hierarchy for the casting function
    sorted_parents = sorted(cls.get_parents(), 
                            key = lambda x: qual_id(sidl.type_id(x)))

    return Template(text).substitute(
        CLASS = iorname, 
        CLASS_LOW = str.lower(iorname),
        CLASS_NAME = '.'.join(cls.qualified_name),
        Casts = cast_binary_search(sorted_parents, cls, True),
        Baseclass = baseclass(cls),
        EPVinits = EPVinits(cls),
        EPVfini = EPVfini(cls),
        EXEC_METHODS = generateMethodExecs(cls, iorname),
        INIT_SEPV = init_sepv(cls, iorname),
        INIT_EPV = init_epv(cls, iorname, sorted_parents),
        GET_EPVS = get_epvs(cls, iorname),
        S_METHODS = s_methods(cls, iorname),
        SET_CONTRACTS = set_contracts(cls, iorname),
        SET_CONTRACTS_STATIC = set_contracts_static(cls, iorname),
        SET_EPV_DECL = set_epv_decl(cls, iorname),
        DUMP_STATS = dump_stats(cls, iorname),
        DUMP_STATS_STATIC = dump_stats_static(cls, iorname),
        ParentDecls = ParentDecls(cls),
        StaticEPVDecls = StaticEPVDecls(sorted_parents, cls, iorname),
        External_getSEPV = (('%s__getStaticEPV,'%iorname) if cls.has_static_methods
                            else '/* no SEPV */'),
        IOR_MAJOR = config.BABEL_VERSION[0], # this will break at Babel 10.0!
        IOR_MINOR = config.BABEL_VERSION[2:],
        CHECK_SKELETONS = check_skeletons(cls, iorname),
        GET_STATIC_EPVS = get_static_epvs(cls, iorname),
        GET_TYPE_STATIC_EPV = get_type_static_epv(cls, iorname),
        GET_ENSURE_LOAD_CALLED = get_ensure_load_called(cls, iorname)
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

${SET_EPV_DECL}

extern void ${CLASS}__call_load(void);
#ifdef __cplusplus
}
#endif

${EXEC_METHODS}

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
${SET_CONTRACTS}
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
${DUMP_STATS}
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
 * HOOKS: Enable/disable static hooks.
 */

static void ior_${CLASS}__set_hooks_static(
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
${DUMP_STATS_STATIC}
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
${SET_CONTRACTS_STATIC}
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

static const char*
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

static void
ior_${CLASS}__exec(
  struct ${CLASS}__object* self,
  const char* methodName,
  struct sidl_rmi_Call__object* inArgs,
  struct sidl_rmi_Return__object* outArgs,
  struct sidl_BaseInterface__object **_ex )
{
  static const struct ${CLASS}__method  s_methods[] = {
${S_METHODS}
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct ${CLASS}__method);
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


${GET_TYPE_STATIC_EPV}
${CHECK_SKELETONS}
${GET_ENSURE_LOAD_CALLED}
${INIT_SEPV}
${INIT_EPV}
${GET_EPVS}

/*
 * __getSuperEPV: returns parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* ${CLASS}__getSuperEPV(void) {
  return s_par_epv__sidl_baseclass;
}

${GET_STATIC_EPVS}

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
      sidl_ClassInfoI_setName(impl, "${CLASS_NAME}", _ex);
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

def cast_binary_search(sorted_types, cls, addref):
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
                  ind+'  cast = %s;' % class_to_interface_ptr(cls, e),
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

def class_to_interface_ptr(cls, e):
    """
    Generate an expression to obtain a pointer to an interface or
    subcls from an object pointer.
   
    @param cls             the object pointer this is a class pointer to
                           this type
    @param e               this is the type of the interface/subclass pointer
                           to be obtained
    @return  a String containing the expression to cast & (if necessary)
             dereference the this pointer to the appropriate internal
             data structure
    """

    @accepts(list, object, tuple)
    def hasAncestor(excluded, search, target):
        #print 'search=%r, target=%r, excluded=%r' %( search.name, target, excluded)
        hsearch = sidl.hashable_type_id(search.data)
        if hsearch in excluded: return False
        if hsearch == target: return True
        for e in search.get_parent_interfaces():
            if hasAncestor(excluded, make_extendable(cls.symbol_table, e), target):
                return True
        return False

    def nextAncestor(ancestor, result):
        ancestor = ancestor.get_parent()
        if ancestor:
            result.append(".d_")
            result.append(qual_cls_low(ancestor))
        return ancestor

    @accepts(object, tuple)
    def directlyImplements(cls, e):
        while cls:
            if sidl.hashable_type_id(e) in cls.get_direct_parent_interfaces():
                return True
            cls = cls.get_parent()
        return False

    @accepts(object, tuple)
    def implementsByInheritance(cls, e):
        parent = cls.get_parent()
        if parent:
            excludedInterfaces = parent.get_parent_interfaces()
        else:
            excludedInterfaces = []

        for ext in cls.get_unique_interfaces():
            ext = make_extendable(*cls.symbol_table[ext])
            if hasAncestor(excludedInterfaces, ext, sidl.hashable_type_id(e)):
                return True
            
        return False

    if (cls.inherits_from(sidl.type_id(e))
        or hasAncestor([], cls, sidl.hashable_type_id(e))): 
      if sidl.is_class(e):
          # fixme: for enums, this is not true
          return '((struct %s__object*)self)'%qual_name(cls.symbol_table, e)
      
      else:
        ancestor = cls
        result = []
        direct = directlyImplements(cls, e)
        result.append('&((*self)')
        while ancestor:
            if ((direct and (sidl.hashable_type_id(e) in ancestor.get_unique_interfaces())) 
                or ((not direct) and implementsByInheritance(ancestor, e))):
                result.append('.d_')
                result.append(qual_name_low(ancestor.symbol_table, e))
                break
            else:
                ancestor = nextAncestor(ancestor, result)

        if ancestor == None:
            raise Exception('Illegal symbol table entry: %s and %s' 
                            % ( cls.name, sidl.type_id(e)))
        
        result.append(')')
        return ''.join(result)
      
    else:
        return 'NULL'
    

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

def qual_cls(cls):
    return '_'.join(cls.qualified_name)

def qual_cls_low(cls):
    return str.lower('_'.join(cls.qualified_name))


def baseclass(cls):
    r = []
    parent = cls.get_parent()
    while parent:
        r.append('.d_'+str.lower('_'.join(parent.qualified_name)))
        parent = parent.get_parent()
    return ''.join(r)


def ParentDecls(cls):
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
        w = len('_'.join(parent.qualified_name))
        if w > width:
            width = w
            
        parent = parent.get_parent()

    def generateParentSelf(cls, level):
        if cls:
            # Now use the width information to print out symbols.
            typ = qual_cls(cls)
            if level == 0:
                r.append('  struct %s__object*'%typ+' '*(width-len(typ))+' s0 = self;')
            else:
                r.append('  struct %s__object*'%typ+' '*(width-len(typ))+' s%d = &s%d->d_%s;' 
                         % (level,level-1,qual_cls_low(cls)))
     
            generateParentSelf(cls.get_parent(), level + 1)

    r = []
    generateParentSelf(cls, 0)
    return '\n'.join(r)

def StaticEPVDecls(sorted_parents, cls, ior_name):
    """
    Collect all the parents of the class in a set and output EPV structures
    for the parents.
    """
    r = []

    # The class
    t = ior_name
    n = str.lower(t)
    if cls.has_static_methods:
        r.append('static void %s__init_sepv(void);'%t)
        r.append('static VAR_UNUSED struct %s__sepv  s_stc_epv__%s;' % (t, n))
        if generateContractEPVs(cls):
            r.append('static VAR_UNUSED struct %s__sepv  s_stc_epv_contracts__%s;' % (t, n))

    # Interfaces and parents
    new_interfaces = cls.get_unique_interfaces()
    for parent in sorted_parents:
        is_par   = not sidl.hashable_type_id(parent) in new_interfaces
        t = qual_id(sidl.type_id(parent), '_')
        n = str.lower(t)
        r.append('static struct %s__epv  s_my_epv__%s;'% (t, n))
        with_parent_hooks = generateHookEPVs(make_extendable(cls.symbol_table, parent))
        if with_parent_hooks:
            r.append('static struct %s__epv  s_my_pre_epv_hooks__%s;'% (t, n))
        if is_par:
          r.append('static struct %s__epv*  s_par_epv__%s;'% (t, n))
          if with_parent_hooks:
              r.append('static struct %s_pre__epv*  s_par_epv_hooks__%s;'% (t, n))
        r.append('')
    
    if cls.has_static_methods and (generateContractEPVs(cls) or generateHookEPVs(cls)):
        r.append('/* Static variables for interface contract enforcement and/or hooks controls. */')
        r.append('static VAR_UNUSED struct %s__cstats s_cstats;' % ior_name)
    
    if generateContractChecks(cls):
        r.append('/* Static file for interface contract enforcement statistics. */')
        r.append('static FILE* s_dump_fptr = NULL;')
        r.append('')
 
    # Declare static hooks epvs
    if generateHookEPVs(cls):
        r.append('static struct %s__pre_epv s_preEPV;'% ior_name)
        r.append('static struct %s__post_epv s_postEPV;'% ior_name)
 
        if cls.has_static_methods:
            r.append('static struct %s__pre_sepv s_preSEPV;'% ior_name)
            r.append('static struct %s__post_sepv s_postSEPV;'% ior_name)

        r.append('')
 
    if False: #fastcall:
        r.append('/* used for initialization of native epv entries. */')
        r.append('static const sidl_babel_native_epv_t NULL_NATIVE_EPV  = { BABEL_LANG_UNDEF, NULL};')

    return '\n'.join(r)

def init_sepv(cls, ior_name):
    """
    generate the sepv initialization function
    """
    if not cls.has_static_methods:
        return ''

    contracts = generateContractEPVs(cls)
    hooks = generateHookEPVs(cls)
    t = ior_name
    n = str.lower(t)
    r = []
    r.append('''
/*
 * SEPV: create the static entry point vector (SEPV).
 */

static void {t}__init_sepv(void)
{{
  struct sidl_BaseInterface__object *throwaway_exception = NULL;
  struct {t}__sepv*  s = &s_stc_epv__{n};'''.format(t=t, n=n))
    if contracts:
        r.append('  struct %s__sepv* cs = &s_stc_epv_contracts__%s;'%(t, n))

    r.append('')
    if hooks:
        r.append('  s->f__set_hooks_static       = ior_%s__set_hooks_static;'%t)

    if contracts:
        r.append('  s->f__set_contracts_static   = ior_%s__set_contracts_static;'%t)
        r.append('  s->f__dump_stats_static      = ior_%s__dump_stats_static;'%t)

    r.append('')
    for m in cls.get_methods():
        r.append('  s->f_%s = NULL;'%sidl.method_id(m))

    r.append('')
    r.append('  %s__set_sepv(s, &s_preSEPV, &s_postSEPV);'%t)
    r.append('')

    if hooks:
        r.append('  ior_%s__set_hooks_static(FALSE, &throwaway_exception);'%t)
        r.append('')

    if contracts:
        r.append('  memcpy((void*)cs, s, sizeof(struct %s__sepv));'%t)
        for m in cls.get_methods():
            n = sidl.method_id(m)
            r.append('  cs->f_%s = check_%s_%s;'%(n, qual_cls(cls), n))

    r.append('')
    r.append('  s_static_initialized = 1;')
    r.append('  ior_%s__ensure_load_called();'%t)
    r.append('}')
    return '\n'.join(r)    


def aliasEPVs(r, T, t, cls, parents):
    """
    Generate the entry point vector alias for each parent class and each
    interface as well as a special one for the current object.
    """
    # Get the width of the symbols for pretty printing.
    width = len(T) + 7
    #w     = Utilities.getWidth(parents) + "struct __epv*".length()

    #if w > width:
    #    width = w

    # Output the EPV pointer for this class and its class and interface
    # parents.
    r.append('  struct %s epv = &s_my_epv__%s;'%(align(T+'__epv*', width), t))
    if generateContractChecks(cls):
        r.append('  struct %s cepv = &s_my_epv_contracts__%s;'%(align(T+'__epv*', width), t))
    
    if generateHookMethods(cls):
        r.append('  struct %s hepv = &s_my_epv_hooks__%s;'%(align(T+'__epv*', width), t))    

    e = 0
    for parent in parents[:-1]:
        par = make_extendable(cls.symbol_table, parent)
        P = '_'.join(par.qualified_name)
        p = str.lower(P)
        r.append('  struct %s e%d = &s_my_epv__%s;'%(align(P+'__epv*', width), e, p))

        if generateHookMethods(par):
            r.append('  struct %s he%d = &s_my_epv_hooks__%s;'%(align(P+'__epv*', width), e, p))    
      
        e += 1
    
    r.append('')
  

  
def generateParentEPVs(r, cls, level, width):
    """
    Recursively output epv pointers to the SIDL objects for this class and its
    parents. The self pointers are of the form sN, where N is an integer
    represented by the level argument. If the width is zero, then the width of
    all parents is generated automatically.
    """
    if cls:
        # Calculate the width of this class and all parents for pretty output.
        # Ooh, very pretty.
        width = 0
        parent = cls
        while parent:
            w = len('_'.join(parent.qualified_name)) + len('struct __epv*')
            if w > width:
                width = w
                
            parent = parent.get_parent()
      
        # Now use the width information to print out symbols.
        # Get rid of the now unused s0 var. 
        if level != 0:
            name = '_'.join(cls.qualified_name)
            r.append('  struct %s s%d = NULL;'%(align(name+'__epv*', width), level))
     
            if generateHookMethods(cls):
                name = '_'.join(cls.qualified_name)
                r.append('  struct %s hs%d = NULL;'%(align(name+'__epv*', width), level))
          
        generateParentEPVs(r, cls.get_parent(), level + 1, width)
    
  

def generateGetParentEPVs(r, T, parent):
    r.append("  // Get my parent's EPVs so I can start with their functions.")

    r.append('  %s__getEPVs('%'_'.join(parent.qualified_name))
    listEPVs(r, parent, True)
    r.append('  );')
    r.append('')
  
def listEPVs(r, cls, first):
    if cls:
        listEPVs(r, cls.get_parent(), False);
        for intf in cls.get_unique_interfaces():
            ifc = make_extendable(cls.symbol_table, intf)
            name = str.lower('_'.join(ifc.qualified_name))
            r.append('    &s_par_epv__%s,'%name)
            if generateHookEPVs(ifc):
                r.append('    &s_par_hepv__%s,'%name)

        name = str.lower('_'.join(cls.qualified_name))
        r.append('    &s_par_epv__%s%s'%(name, ',' if not first or generateHookEPVs(cls) else ''))
        if generateHookEPVs(cls):
            r.append('    &s_par_hepv__%s%s'%(name, ',' if not first else ''))


def saveEPVs(r, cls, level):
    """
    Recursively save the class and interface EPVs in this class and all parent
    classes. Nothing is done if the class argument is null.
    """

    if cls:
        saveEPVs(r, cls.get_parent(), level + 1)

        # Save the class entry point vector.
        name = str.lower('_'.join(cls.qualified_name))
        r.append('  s%d = s_par_epv__%s;'%(level, name))

        if generateHookMethods(cls):
            r.append('  hs%d = s_par_hepv__%s;'%(level, name))

def generateEPVMethodAssignments(r, ext, setType, ptrVar, doStatics):
    """
    Generate the appropriate contract or hooks methods assignments.
    """
    methods = ext.local_static_methods if doStatics else ext.local_nonstatic_methods
    #int      mwidth  = Math.max(Utilities.getWidth(methods), s_longestBuiltin)
    #                 + IOR.getVectorEntry("").length() + ptrVar.length() + 2
    doChecks = setType == 'set_contracts'

    hasInvs = ext.has_inv_clause(True)

    for m in methods:
        if (not doChecks) or (hasInvs or m.hasPreClause() or m.hasPostClause()):
            tName = 'hooks_'+sidl.long_method_name(m) if doChecks else 'check_'+sidl.long_method_name(m)
            r.append(' %s->f_%s = %s;' % (ptrVar, sidl.long_method_name(m), tName))
      

def generateNonstaticMethods(r, cls, obj, methods, var, parent, mwidth):
    """
    Generate the non-static methods for the specified class.
    """
    # Iterate through all of the nonstatic methods. Assign them to NULL if the
    # parent class does not have the method and to the parent function pointer
    # if the parent has the method.
    for method in methods:
        mname     = sidl.long_method_name(method)
        parentHas = parent and parent.has_method_by_long_name(mname, _all=True)

        generateNonstaticEPV(r, method, obj, var, parent, parentHas, mwidth)
        
        #if (method.getCommunicationModifier() == Method.NONBLOCKING
        #    and not skip_rmi):
        #    send = method.spawnNonblockingSend()
        #    if  send: 
        #        mname     = sidl.long_method_name(send()
        #        parentHas = parent and parent.has_method_by_long_name(mname, true)
        #        assignNonblockingPointer(r, cls, send, mwidth); 
        #    
        #    recv = method.spawnNonblockingRecv()
        #    if  recv != null : 
        #        mname     = sidl.long_method_name(recv()
        #        parentHas = parent and parent.has_method_by_long_name(mname, true)
        #        assignNonblockingPointer(r, cls, recv, mwidth); 
            
      
    r.append('')

    t = '_'.join(cls.qualified_name)
    # Call the user initialization function to set up the EPV structure.
    if generateHookEPVs(cls):
        r.append('  %s__set_epv(%s, &s_preEPV, &s_postEPV);'%(t, var))
    else:
        r.append('  %s__set_epv(%s);'%(t, var))
    
    # Now, if necessary, clone the static that has been set up and overlay with
    # local methods for contracts and/or hooks.
    r.append('')
    addBlank = False
    if generateContractChecks(cls):
        xEPV = "cepv"
        r.append("  memcpy((void*)%s, %s, sizeof(struct %s__epv));"%(xEPV, var, t))
        generateEPVMethodAssignments(r, cls, 'set_contracts', xEPV, False)
        addBlank = True
    

    if generateHookMethods(cls):
        xEPV = "hepv"
        r.append("  memcpy((void*)%s, %s, sizeof(struct %s__epv));"%(xEPV, var, t))
        if addBlank:
            r.append('')
            addBlank = False
      
        generateEPVMethodAssignments(r, cls, 'set_hooks', xEPV, False)
        addBlank = True
    
    if addBlank:
        r.append('')
    
def assignNonblockingPointer(r, cls, method, mwidth):
    """
    Generate a nonblocking single assignment in initialization of the EPV 
    pointer.
  
    TODO: This kludge should probably be removed when we figure out how we
    want to do nonblocking for local objects permanently.  This just
    generates an exception.
  
    @param name
           String name of the extendable
    @param method
           The method in the EPV to initialzie
    @param mwidth
           formatting parameter related to width of longest method name
    """

    mname = sidl.long_method_name(method)
    r.append('  epv->f_%s= ior_%s_%s;'(align(mname, mwidth), '_'.join(cls.qualified_name), mname))
  

def generateNonstaticEPV(r, m, obj, var, parent, parentHas, width):
    """
    Generate the non-static method epv entry for the method.
    """
    name = sidl.long_method_name(m)
    if parentHas:
        r.append('  %s->f_%s= %ss1->f_%s;'%(var, align(name, width), 
                                            getCast(parent.symbol_table, m, obj + "*"), name))
    else:
        r.append('  %s->f_%s= NULL;'%(var, align(name, width)))
    
    #if d_context.getConfig().getFastCall() && !IOR.isBuiltinMethod(name):
    #    if parentHas:
    #        guard = IOR.getNativeEPVGuard(parent)
    #        r.append("#ifdef " + guard)
    #        r.append(var + "->")
    #        r.append(IOR.getNativeVectorEntry(name), width)
    #        r.append(" = s1->")
    #        r.append(IOR.getNativeVectorEntry(name) + ";")
    #        r.append("#else")
    #        r.append(var + "->")
    #        r.append(IOR.getNativeVectorEntry(name), width)
    #        r.append(" = NULL_NATIVE_EPV;")
    #        r.append("#endif /*" + guard + "*/")
    #    else:
    #        r.append('%s->%s%s= NULL_NATIVE_EPV;'%(var,IOR.getNativeVectorEntry(name), ' '*width))

def align(text, length):
    return text+' '*(length-len(text))

def copyEPVs(r, symbol_table, parents, renames):
    """
    Copy EPV function pointers from the most derived EPV data structure into
    all parent class and interface EPVs.
    """

    e = 0;
    for ext1 in parents[:-1]:
        # Extract information about the parent extendable object. Generate a list
        # of the nonstatic methods and calculate the width of every method name.
        # ext = (Extendable) Utilities.lookupSymbol(d_context, id);
        ext = make_extendable(symbol_table, ext1)
        ext.scan_methods()

        methods = ext.all_nonstatic_methods
        #mwidth = Math.max(Utilities.getWidth(methods), s_longestBuiltin)
        #           + IOR.getVectorEntry("").length();
        mwidth = 16
        name = '_'.join(ext.qualified_name)
   
        # Calculate the "self" pointer for the cast function.
        if ext.is_interface():
          selfptr = "void*";
        else:
          selfptr = 'struct %s__object*'%name;
   
        # Generate the assignments for the cast and delete functions.
        estring  = 'e%d->'%e
        vecEntry = '_cast'
        ex = 'struct sidl_BaseInterface__object**'
   
        r.append('  // Override function pointers for %s with mine, as needed.'%name)
        r.append('  %sf_%s= (void* (*)(%s,const char*, %s))epv->f_%s;'
                 %(estring,align(vecEntry, mwidth),selfptr,ex,vecEntry))
   
        vecEntry = '_delete'
        r.append('  %sf_%s= (void* (*)(%s,%s))epv->f_%s;'
                 %(estring,align(vecEntry, mwidth),selfptr,ex,vecEntry))
   
        if not skip_rmi:
   
          vecEntry = '_getURL'
          r.append('  %sf_%s= (char* (*)(%s,%s))epv->f_%s;'
                 %(estring,align(vecEntry, mwidth),selfptr,ex,vecEntry))
   
          vecEntry = '_raddRef'
          r.append('  %sf_%s= (void (*)(%s,%s))epv->f_%s;'
                 %(estring,align(vecEntry, mwidth),selfptr,ex,vecEntry))
   
          vecEntry = '_isRemote'
          r.append('  %sf_%s= (sidl_bool* (*)(%s,%s))epv->f_%s;'
                 %(estring,align(vecEntry, mwidth),selfptr,ex,vecEntry))
   
          # Generate the assignment for exec function
          vecEntry = '_exec'
          r.append('  %sf_%s= (void (*)(%s,const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,%s))epv->f_%s;'
                 %(estring,align(vecEntry, mwidth),selfptr,ex,vecEntry))
   
          methList = ext.all_nonblocking_methods
        
        else:
          methList = ext.all_nonstatic_methods
   
        T = '_'.join(ext.qualified_name)
        # Iterate over all methods in the EPV and set the method pointer.
        for method in methList:
          name          = T+sidl.long_method_name(method)
          oldname       = sidl.long_method_name(method)
          if sidl.long_method_name(method) in renames:
            newname = sidl.long_method_name(renames[method])
          else:
            newname = oldname
   
          r.append('  %sf_%s= %sepv->f_%s;'
                 %(estring,align(oldname, mwidth),getCast(symbol_table, method, selfptr),newname))
   
   
        # Do the same for the native EPV entries if enabled
        r.append('')
        # if fastcall:
        #   String guard = IOR.getNativeEPVGuard(ext);
        #   r.appendlnUnformatted("#ifdef " + guard);
        #  
        #   for (Iterator j = methList.iterator(); j.hasNext(); ) {
        #     Method method        = (Method) j.next();
        #     String name          = id.getFullName() + ".";
        #     name                += sidl.long_method_name(method();
        #     Method renamedMethod = (Method) renames.get(name);
        #     String oldname       = sidl.long_method_name(method();
        #     String newname       = null;
        #     if (renamedMethod != null) {
        #       newname = sidl.long_method_name(renamedMethod();
        #     } else { 
        #       newname = sidl.long_method_name(method();
        #     }
        #     r.append(estring);
        #     r.appendAligned(IOR.getNativeVectorEntry(oldname), mwidth);
        #     r.appendln(" = epv->" + IOR.getNativeVectorEntry(newname) + ";");
        #   }
        #   
        #   r.appendlnUnformatted("#endif /*" + guard + "*/");
        # }
        
        # JIM add in he stuff here?
        r.append('');
        if generateHookMethods(ext):
          r.append("  memcpy((void*) he%d, e%d, sizeof(struct %s__epv));"%(e, e, T))
   
          # Iterate over all methods in the EPV and set the method pointer.
          # if we're using hooks
          for method in methods:
              name          = T+'.'+sidl.long_method_name(method)
              oldname       = sidl.long_method_name(method)
              if sidl.method_method_name(method) in renames:
                  newname = sidl.long_method_name(renames[method])
              else:
                  newname = oldname
              r.append('  %she_%s= %shepv->%s;'
                       %(estring,align(vecEntry, mwidth),
                         getCast(symbol_table, method, selfptr),vecEntry))
   
        r.append('');
        e += 1;

def getCast(symbol_table, method, selfptr):
    """
    Generate a cast string for the specified method.  The string
    argument self represents the name of the object.  A code generation
    exception is thrown if any of the required symbols do not exist in
    the symbol table.
    """
    # Begin the cast string with the return type and self object reference.
    cast = []
    cast.append("(");
    cast.append(c_gen(babel.lower_ir(symbol_table, sidl.method_type_void(method))))
    cast.append(" (*)(")

    # Add the method arguments to the cast clause as well as an
    # optional exception argument.
    args = [selfptr]
    for arg in sidl.method_args(method):
        args.append(c_gen(babel.lower_ir(symbol_table, sidl.arg_type_void(arg))))

    args.append('struct sidl_BaseInterface__object **') # exception
    cast.append(', '.join(args))
    cast.append("))")
    return ''.join(cast)
   



def init_epv(cls, iorname, sorted_parents):
    """
    Generate the function that initializes the method entry point vector for
    this class. This class performs three functions. First, it saves the EPVs
    from the parent classes and interfaces (assuming there is a parent).
    Second, it fills the EPV for this class using information from the parent
    (if there is a parent, and NULL otherwise). It then calls the skeleton
    method initialization function for the EPV. Third, this function resets
    interface EPV pointers and parent EPV pointers to use the correct function
    pointers from the class EPV structure.
    """
    T = iorname
    t = str.lower(T)
    r = []
    r.append('// EPV: create method entry point vector (EPV) structure.')

    # Declare the method signature and open the implementation body.
    r.append("static void %s__init_epv(void)"%T)

    r.append('{')
    r.append('  // assert( " + IOR.getHaveLockStaticGlobalsMacroName() + " );')

    # Output entry point vectors aliases for each parent class and interface as
    # well as a special one for the current object.
    parents = sorted_parents
    aliasEPVs(r, T, t, cls, parents)

    # Output pointers to the parent classes to simplify access to their data
    # structures.
    if cls.get_parent:
        generateParentEPVs(r, cls, 0, 0)
        r.append('')
    
    # Get the parent class EPVs so our versions start with theirs.
    parent = cls.get_parent()
    # FIXME: inefficient!
    parent.scan_methods()
    if parent:
      generateGetParentEPVs(r, T, parent)
      
      # Save all parent interface and class EPVs in static pointers.
      r.append('')
      r.append('  // Alias the static epvs to some handy small names.')
      saveEPVs(r, parent, 1)
      r.append('')

    # Generate a list of the nonstatic methods in the class and get the width
    # of every method name.
    methods = cls.all_nonstatic_methods
    #mwidth  = max(Utilities.getWidth(methods), s_longestBuiltin)
    #               + IOR.getVectorEntry("").length()
    mwidth = 16

    # Output builtin methods.
    for m in babel.builtin_method_names:
        if (m <> '_load') and not (m in babel.rmi_related and skip_rmi):
            mname = 'NULL' if m in ['_ctor', '_ctor2', '_dtor'] else 'ior_%s_%s'%(T, m)
            r.append('  epv->f_%s= %s;'%(align(m, 16), mname))

    # Output the class methods.
    generateNonstaticMethods(r, cls, 'struct %s__object'%T, methods, "epv", parent, mwidth)

    # Iterate through all parent EPVs and set each of the function pointers to
    # use the corresponding function in the parent EPV structure.
    renames = dict()
    resolveRenamedMethods(cls, renames)
    copyEPVs(r, cls.symbol_table, parents, renames)
    r.append("  s_method_initialized = 1;")
    r.append("  ior_" + T + "__ensure_load_called();")

    # Close the function definition.
    r.append("}")
    r.append('')
    return '\n'.join(r)

def get_epvs(cls, iorname):
    """
    Generate method to return my version of the relevant EPVs -- 
    mine and all of my ancestors.
   
    The purpose of this method is to make my version of the
    EPVs available to my immediate descendants (so they can
    use my functions in their versions of their EPVs).
    """
    T = iorname
    r = []
    r.append("/**")
    r.append("* %s__getEPVs: Get my version of all relevant EPVs."%T)
    r.append("*/")
    r.append("void %s__getEPVs("%T)
    declareEPVsAsArgs(r, cls, True)
    r.append("  )")
    r.append("{")

    # Ensure that the EPV has been initialized.
    r.append("  LOCK_STATIC_GLOBALS;")
    r.append("  if (!s_method_initialized) {")
    r.append("    %s__init_epv();"%T)
    r.append("  }")
    r.append("  UNLOCK_STATIC_GLOBALS;")
    r.append('')

    setEPVsInGetEPVs(r, cls)
    r.append("}")
        
    return '\n'.join(r)

def declareEPVsAsArgs(r, cls, first):
    if cls:
        declareEPVsAsArgs(r, cls.get_parent(), False)
        for ifce in sorted(cls.get_unique_interfaces()):
            ifc = make_extendable(cls.symbol_table, ifce)
            T = '_'.join(ifc.qualified_name)
            t = str.lower(T)
            r.append('  struct %s__epv **s_arg_epv__%s,'%(T, t))
            if generateHookEPVs(ifc):
                r.append('  struct %s__epv **s_arg_epv_hooks__%s,'%(T, t))
        T = '_'.join(cls.qualified_name)
        t = str.lower(T)
        hooks = generateHookEPVs(cls)
        r.append('  struct %s__epv **s_arg_epv__%s'%(T, t)+(',' if (not first) or hooks else ''))
        if hooks:
            r.append('  struct %s__epv **s_arg_epv_hooks__%s'%(T, t)+(',' if not first else ''))


def setEPVsInGetEPVs(r, cls):
    if cls:
        setEPVsInGetEPVs(r, cls.get_parent())
        for ifce in sorted(cls.get_unique_interfaces()):
            ifc = make_extendable(cls.symbol_table, ifce)
            t = str.lower('_'.join(ifc.qualified_name))
            r.append('  *s_arg_epv__%s = &s_my_epv__%s;'%(t, t))
            if generateHookEPVs(ifc):
                r.append('  *s_arg_epv_hooks__%s = &s_my_epv_hooks__%s;'%(t, t))

        t = str.lower('_'.join(cls.qualified_name))
        r.append('  *s_arg_epv__%s = &s_my_epv__%s;'%(t, t))
        if generateHookEPVs(cls):
            r.append('  *s_arg_epv_hooks__%s = &s_my_epv_hooks__%s;'%(t, t))


def s_methods(cls, iorname):
    r = []
    for m in cls.all_nonstatic_methods:  
        ln = sidl.long_method_name(m)
        r.append('    { "%s", %s_%s__exec },'%(ln, iorname, ln));
    return '\n'.join(r)

def set_contracts(cls, ior_name):
    return '  /* empty because contract checks not needed */'

def dump_stats(cls, ior_name):
    return '  /* empty because contract checks not needed */'

def dump_stats_static(cls, ior_name):
    if cls.has_static_methods and generateContractEPVs(cls):
        return Template(r'''
    struct ${t}__method_cstats *ms;

    int         i;
    sidl_bool   firstTime = FALSE;
    sidl_bool   reported  = FALSE;
    const char* fname     = (filename) ? filename : "ContractStats.out";

    if (s_dump_fptr == NULL) {
      firstTime = TRUE;
      if ((s_dump_fptr=fopen(fname,"w")) == NULL) {
        fprintf(stderr, "Cannot open file %s to dump the static interface contract enforcement statistics.\n", fname);
        return;
      }
    }

    if (firstTime) {
      sidl_Enforcer_dumpStatsHeader(s_dump_fptr, FALSE);
      fprintf(s_dump_fptr, "; Method; Checked; Okay; Violated; MethExcepts\n\n");
    }

    for (i = s_IOR_${T}_MIN;
         i<= s_IOR_${T}_MAX; i++) {
      ms = &s_cstats.method_cstats[i];
      if (  (s_ior_${t}_method[i].is_static) 
         && (ms->tries > 0) ) {
        reported = TRUE;
        sidl_Enforcer_dumpStatsData(s_dump_fptr, prefix, FALSE);
        fprintf(s_dump_fptr, "; %s; %d; %d; %d; %d\n",
            s_ior_${t}_method[i].name,
            ms->tries,
            ms->successes,
            ms->failures,
            ms->nonvio_exceptions);
      }
    }

    if (reported) {
      fprintf(s_dump_fptr, "\n");
    } else {
      sidl_Enforcer_dumpStatsData(s_dump_fptr, prefix, FALSE);
      fprintf(s_dump_fptr, "; No attempts to enforce contracts detected\n\n");
    }

    fflush(s_dump_fptr);
    return;
''').substitute(t = ior_name, T = str.upper(ior_name))
    else:
        return '  /* empty since there are no static contracts */'


def set_contracts_static(cls, ior_name):
    if cls.has_static_methods and generateContractEPVs(cls):
        return Template(r'''
  {
    struct ${t}__method_desc *md;
    struct ${t}__method_cstats *ms;

    const char* filename = (enfFilename) ? enfFilename
                         : "${t}.dat";
    FILE*  fptr   = NULL;
    int    ind, invc, prec, posc;
    double invt, mt, pret, post;

    s_cstats.enabled = enable;

    if (  (filename) 
       && (sidl_Enforcer_usingTimingData()) ) {
      fptr = fopen(filename, "r");
      if (fptr != NULL) {
        /*
         *  * The first line is assumed to contain the invariant
         *  * complexity and average enforcement cost REGARDLESS
         *  * of specification of invariants.
         */

        fscanf(fptr, "%d %lf\n", &invc, &invt);
        while (fscanf(fptr, "%d %d %d %lf %lf %lf\n",
          &ind, &prec, &posc, &mt, &pret, &post) != EOF)
        {
          if (  (s_IOR_${T}_MIN <= ind)
             && (ind <= s_IOR_${T}_MAX) ) {
            md = &s_ior_${t}_method[ind];
            md->pre_complexity  = prec;
            md->post_complexity = posc;
            md->meth_exec_time  = mt;
            md->pre_exec_time   = pret;
            md->post_exec_time  = post;
          } else {
            fprintf(stderr, "ERROR:  Invalid method index, %d, in contract metrics file %s\n", ind, filename);
            return;
          }
        }
        fclose(fptr);
      }
    }

    if (resetCounters) {
      int i;
      for (i =s_IOR_${T}_MIN;
           i<=s_IOR_${T}_MAX; i++) {
        ms = &s_cstats.method_cstats[i];
        ms->tries          = 0;
        ms->successes      = 0;
        ms->failures       = 0;
        ms->nonvio_exceptions = 0;

        md = &s_ior_${t}_method[i];
        RESETCD(md);
      }
    }
  }
''').substitute(t = ior_name, T = str.upper(ior_name))
    else:
        return '  /* empty since there are no static contracts */'


def set_epv_decl(cls, ior_name):
    if generateHookEPVs(cls):
        r = '''
extern void ${CLASS}__set_epv(
  struct ${CLASS}__epv* epv,
    struct ${CLASS}__pre_epv* pre_epv,
    struct ${CLASS}__post_epv* post_epv);
'''
    else:
        r = 'extern void ${CLASS}__set_epv( struct ${CLASS}__epv* epv );'
    return Template(r).substitute(CLASS=ior_name)
    

builtin_funcs = {
    'allBoth'   : 'All_Both',
    'allVl'     : 'ALL_VL', 
    'allVr'     : 'ALL_VR', 
    'anyBoth'   : 'ANY_BOTH', 
    'anyVl'     : 'ANY_VL', 
    'anyVr'     : 'ANY_VR', 
    'countBoth' : 'COUNT_BOTH', 
    'countVl'   : 'COUNT_VL', 
    'countVr'   : 'COUNT_VR', 
    'dimen'     : 'DIMEN', 
    'irange'    : 'IRANGE', 
    'lower'     : 'LOWER', 
    'max'       : 'MAX', 
    'min'       : 'MIN', 
    'nearEqual' : 'NEAR_EQUAL', 
    'nonIncr'   : 'NON_INCR', 
    'noneBoth'  : 'NONE_BOTH', 
    'noneVl'    : 'NONE_VL', 
    'noneVr'    : 'NONE_VR', 
    'range'     : 'RANGE', 
    'size'      : 'SIZE', 
    'stride'    : 'STRIDE', 
    'sum'       : 'SUM', 
    'nonDecr'   : 'NON_DECR', 
    'upper'     : 'UPPER',
    'irange'    : 'IRANGE', 
    'nearEqual' : 'NEAR_EQUAL',
    'range'     : 'RANGE'
}

@matcher(globals())
def lower_assertion(cls, m, expr):
    """
    convert a SIDL assertion expression into IR code
    """
    def low(e): 
        return lower_assertion(cls, m, e)

    def get_arg_type(name):
        if name == '_retval':
            return m[1]

        args = sidl.method_args(m)
        for arg in args:
            if sidl.arg_id(arg) == name:
                return arg[3]
        raise Exception('arg not found: '+name)


    with match(expr):
        if (sidl.infix_expr, sidl.iff, Lhs, Rhs):
            return ir.Infix_expr(ir.log_or,
                                 ir.Infix_expr(ir.log_and, low(Lhs), low(Rhs)),
                                 ir.Infix_expr(ir.log_and,
                                               ir.Prefix_expr(ir.log_not, low(Lhs)), 
                                               ir.Prefix_expr(ir.log_not, low(Rhs))))

        if (sidl.infix_expr, sidl.implies, Lhs, Rhs):
            return ir.Infix_expr(ir.log_or, ir.Prefix_expr(ir.log_not, low(Lhs)), low(Rhs))

        elif (sidl.infix_expr, Bin_op, Lhs, Rhs):
            return ir.Infix_expr(Bin_op, low(Lhs), low(Rhs))

        elif (sidl.prefix_expr, Un_op, AssertExpr):
            return ir.Prefix_expr(Un_op, AssertExpr)

        elif (sidl.fn_eval, Id, AssertExprs):
            args = [low(e) for e in AssertExprs]
            if Id in builtin_funcs.keys():
                try: arg0 = AssertExprs[0]
                except: print "**ERROR: assert function has no arguments: ", sidl_gen(expr)
                t = get_arg_type(low(arg0))
                if sidl.is_array(t):
                    typearg = c_gen(babel.lower_ir(cls.symbol_table, t))
                    return ir.Call('SIDL_ARRAY_'+builtin_funcs[Id], [typearg]+args)
                else:
                    return ir.Call('SIDL_'+builtin_funcs[Id], args)
            else:
                return ir.Call('(sepv->f_%s)'%Id, args+['_ex'])

        elif (sidl.var_ref, Id):
            return Id
        else: 
            if expr == 'RESULT':
                return '_retval'
            else: return expr


def precondition_check(cls, m, assertion):
    """
    convert a SIDL assertion expression into IR code and return a string with a check
    """
    _, name, expr = assertion
    t = '.'.join(cls.qualified_name)
    a = sidl_gen(assertion)
    ac = c_gen(lower_assertion(cls, m, expr))
    return Template(r'''if (!(${ac})) {
        cOkay  = 0;
        if ((*_ex) == NULL) {
          pre_err = sidl_PreViolation__create(&tae);
          sidl_PreViolation_setNote(pre_err,
            "REQUIRE VIOLATION $n: $t: $m: $a.", 
            &tae);
          (*_ex) = sidl_BaseInterface__cast(pre_err, &tae);
          sidl_PreViolation_deleteRef(pre_err, &tae);
        }
      }''').substitute(t = t,
                       m = sidl.method_id(m), 
                       n = name, 
                       ac = ac, 
                       a = a)

def postcondition_check(cls, m, assertion):
    """
    convert a SIDL assertion expression into IR code and return a string with a check
    """
    _, name, expr = assertion
    t = '.'.join(cls.qualified_name)

    if sidl.is_prefix_expr(expr) and expr[1] == sidl.is_:
        return 'if (NULL) { /* pure */ }'

    a = sidl_gen(assertion)
    ac = c_gen(lower_assertion(cls, m, expr))
    return Template(r'''if (!(${ac})) {
        cOkay  = 0;
        if ((*_ex) == NULL) {
          post_err = sidl_PostViolation__create(&tae);
          sidl_PostViolation_setNote(post_err,
            "ENSURE VIOLATION $n: $t: $m: $a.", 
            &tae);
          (*_ex) = sidl_BaseInterface__cast(post_err, &tae);
          sidl_PostViolation_deleteRef(post_err, &tae);
        }
      }''').substitute(t = t,
                       m = sidl.method_id(m), 
                       n = name, 
                       ac = ac, 
                       a = a)
 
def get_static_epvs(cls, ior_name):
    if not cls.has_static_methods:
        return '/* no get_static_epv since there are no static methods */'

    substs = { 'c': ior_name, 't': str.lower(ior_name) }
    r = Template(r'''
/*
 * ${c}__getStaticEPV: return pointer to static EPV structure.
 */
struct ${c}__sepv*
${c}__getStaticEPV(void){
  struct ${c}__sepv* sepv;
  LOCK_STATIC_GLOBALS;
  if (!s_static_initialized) {
    ${c}__init_sepv();
  }
  UNLOCK_STATIC_GLOBALS;
''').substitute(substs)
    if generateContractEPVs(cls):
        r += Template(r'''
  if (sidl_Enforcer_areEnforcing()) {
    if (!s_cstats.enabled) {
      struct sidl_BaseInterface__object *tae;
      ior_${c}__set_contracts_static(sidl_Enforcer_areEnforcing(),
        NULL, TRUE, &tae);
    }
    sepv = &s_stc_epv_contracts__${t};
  } else 
''').substitute(substs)
    r += Template(r'''{
    sepv = &s_stc_epv__${t};
  }
  return sepv;
}
''').substitute(substs)
    return r


def get_ensure_load_called(cls, ior_name):
    r = Template(r'''
static void ior_${c}__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    s_load_called=1;
    ${c}__call_load();
''').substitute(c=ior_name)
    if generateContractEPVs(cls) and cls.has_static_methods:
        r += Template(r'''
    struct sidl_BaseInterface__object *tae;
    ior_${c}__set_contracts_static(sidl_Enforcer_areEnforcing(), 
      NULL, TRUE, &tae);
''').substitute(c=ior_name)
    r += Template(r'''
  }
}
''').substitute(c=ior_name)
    return r

    
def get_type_static_epv(cls, ior_name):
    if not (generateContractEPVs(cls) and cls.has_static_methods):
        return '/* no type_static_epv since there are no static contracts */'
    else: return Template(r'''
/*
 * ${c}__getTypeStaticEPV: return pointer to specified static EPV structure.
 */

struct ${c}__sepv*
${c}__getTypeStaticEPV(int type){
  struct ${c}__sepv* sepv;
  LOCK_STATIC_GLOBALS;
  if (!s_static_initialized) {
    ${c}__init_sepv();
  }
  UNLOCK_STATIC_GLOBALS;

  if (type == s_SEPV_${T}_CONTRACTS) {
    sepv = &s_stc_epv_contracts__${t};
  } else {
    sepv = &s_stc_epv__${t};
  }
  return sepv;
}
''').substitute(c = ior_name, t = str.lower(ior_name), T = str.upper(ior_name))
      

def check_skeletons(cls, ior_name):
    from chapel.backend import babel_epv_args
    if not generateContractChecks(cls):
        return '  /* no check_* stubs since there are no contracts */'

    r = []
    for m in cls.get_methods():
        (_, Type, _, Attrs, Args, _, _, Requires, Ensures, _) = m
        static = member_chk(sidl.static, Attrs)
        method_name = sidl.method_id(m)
        preconditions = Requires
        postconditions = Ensures
        ctype = c_gen(babel.lower_ir(cls.symbol_table, Type))
        cargs =  babel_epv_args(Attrs, Args, cls.symbol_table, '_'.join(cls.qualified_name))
        
        substs = { 't' : ior_name,    'T' : str.upper(ior_name), 
                   'm' : method_name, 'M' : str.upper(method_name) }
        r.append('static %s check_%s_%s(%s)'%(
                ctype,
                ior_name,
                method_name,
                ', '.join([c_gen(a) for a in cargs])))
        r.append('{')

        if Type <> sidl.void:
            r.append('  %s _retval = 0;'%ctype)

        r.append(Template(r'''
  struct ${t}__sepv* sepv = ${t}__getTypeStaticEPV(s_SEPV_${T}_BASE);
  int     cOkay   = 1;
  double  methAvg = 0.0;
  double  cAvg    = 0.0;
  int     cComp   = 0;

  struct sidl_BaseInterface__object *tae = NULL;

  char*   cName = "${t}";

  struct sidl_PreViolation__object *pre_err;
  struct sidl_PostViolation__object *post_err;
  struct timeval ts0, ts1, ts2, ts3;

  struct ${t}__method_cstats *ms = 
    &s_cstats.method_cstats[s_IOR_${T}_${M}];
  struct ${t}__method_desc *md = 
    &s_ior_${t}_method[s_IOR_${T}_${M}];
  (*_ex)  = NULL;

#ifdef SIDL_CONTRACTS_DEBUG
  printf("check_${t}_${m}: Entered\n");
#endif /* SIDL_CONTRACTS_DEBUG */

  if (md->est_interval > 1) {
    gettimeofday(&ts0, NULL);
  }

  methAvg = md->meth_exec_time;
  cAvg    = md->pre_exec_time;
  cComp   = md->pre_complexity;
''').substitute(substs))

        if preconditions:
            r.append(Template(r'''
#ifdef SIDL_CONTRACTS_DEBUG
  printf("...Precondition: enforceClause=%d\n", 
    sidl_Enforcer_enforceClause(TRUE, sidl_ClauseType_PRECONDITION, cComp, 
      TRUE, FALSE, methAvg, cAvg));
#endif /* SIDL_CONTRACTS_DEBUG */
  if (sidl_Enforcer_enforceClause(TRUE, sidl_ClauseType_PRECONDITION, cComp, 
    TRUE, FALSE, methAvg, cAvg)) {
    (ms->tries) += 1;
''').substitute(substs))

            # all precondition checks
            r.append('    '+precondition_check(cls, m, preconditions[0][1]))
            for _, c in preconditions[1:]:
                r.append('    else '+precondition_check(cls, m, c))

            r.append(r'''
      SIDL_INCR_IF_THEN(cOkay,ms->successes,ms->failures)
    }
''')
        # end if preconditions
    
        r.append(r'''
#ifdef SIDL_NO_DISPATCH_ON_VIOLATION
  if (cOkay) {
#endif /* SIDL_NO_DISPATCH_ON_VIOLATION */
''')
        # the method call
        r.append('    %s(%sepv->f_%s)(%s, _ex);'%(
                '_retval = ' if Type <> sidl.void else '',
                's' if static else '',
                method_name,
                ', '.join([sidl.arg_id(arg) for arg in Args])))
        
        r.append(Template(r'''
      if ((*_ex) != NULL) {
        (ms->nonvio_exceptions) += 1;

      if (md->est_interval > 1)
        gettimeofday(&ts2, NULL);
      else (md->est_interval) -= 1;

      if (!sidl_Enforcer_areTracing()) {
        md->pre_exec_time = SIDL_DIFF_MICROSECONDS(ts1, ts0);
        md->meth_exec_time = SIDL_DIFF_MICROSECONDS(ts2, ts1);
        md->post_exec_time = 0.0;
      } else {
        TRACE(cName, md, s_IOR_${T}_${M}, 0, 0, 0, 
          SIDL_DIFF_MICROSECONDS(ts2, ts1), SIDL_DIFF_MICROSECONDS(ts1, ts0), 
          0.0, 0.0, 0.0);
      }

#ifdef SIDL_CONTRACTS_DEBUG
      printf("...Exiting due to base call exception\n");
#endif /* SIDL_CONTRACTS_DEBUG */

      return _retval;
    }

#ifdef SIDL_NO_DISPATCH_ON_VIOLATION
  }
#endif /* SIDL_NO_DISPATCH_ON_VIOLATION */
''').substitute(substs))

        if postconditions:
            r.append(Template(r'''
  if (md->est_interval > 1) {
    gettimeofday(&ts2, NULL);
  }

  methAvg = md->meth_exec_time;
  cAvg    = md->post_exec_time;
  cComp   = md->post_complexity;
#ifdef SIDL_CONTRACTS_DEBUG
  printf("...Postcondition: enforceClause=%d\n", 
    sidl_Enforcer_enforceClause(TRUE, sidl_ClauseType_POSTCONDITION, cComp, 
      TRUE, FALSE, methAvg, cAvg));
#endif /* SIDL_CONTRACTS_DEBUG */
  if (cOkay && sidl_Enforcer_enforceClause(TRUE, sidl_ClauseType_POSTCONDITION, cComp, 
    TRUE, FALSE, methAvg, cAvg)) {
    (ms->tries) += 1;
''').substitute(substs))

            # all postcondition checks
            r.append(postcondition_check(cls, m, postconditions[0][1]))
            for _, c in postconditions[1:]:
                r.append('    else '+postcondition_check(cls, m, c))

            r.append(r'''
      SIDL_INCR_IF_THEN(cOkay,ms->successes,ms->failures)
    }
''')
        # end if postconditions


        r.append(Template(r'''
  if (md->est_interval > 1)
    gettimeofday(&ts2, NULL);
  else (md->est_interval) -= 1;

  if (!sidl_Enforcer_areTracing()) {
    md->pre_exec_time = SIDL_DIFF_MICROSECONDS(ts1, ts0);
    md->meth_exec_time = SIDL_DIFF_MICROSECONDS(ts2, ts1);
    md->post_exec_time = SIDL_DIFF_MICROSECONDS(ts3, ts2);
  } else {
      TRACE(cName, md, s_IOR_${T}_VUDOT, 0, 0, 0, SIDL_DIFF_MICROSECONDS(
        ts2, ts1), SIDL_DIFF_MICROSECONDS(ts1, ts0), SIDL_DIFF_MICROSECONDS(ts3,
        ts2), 0.0, 0.0);
  }
  RESETCD(md)

#ifdef SIDL_CONTRACTS_DEBUG
  printf("check_${t}_${m}: Exiting normally\n");
#endif /* SIDL_CONTRACTS_DEBUG */
''').substitute(substs))
        if Type <> sidl.void:
            r.append('  return _retval;')
        
        r.append('}')
        # end for each method

    return '\n'.join(r)


def contract_decls(cls, iorname):
    if not generateContractEPVs(cls):
        return []
    r = [Template('''
/*
 * Define invariant clause data for interface contract enforcement.
 */

static VAR_UNUSED struct ${t}__inv_desc{
  int    inv_complexity;
  double inv_exec_time;
} s_ior_${t}_inv = {
  0, 0.0,
};

/*
 * Define method description data for interface contract enforcement.
 */
''').substitute(t=iorname)]
    n = 0
    T = str.upper(iorname)
    r.append('static const int32_t s_IOR_%s_MIN = 0;'%T)
    for m in cls.get_methods():
        r.append('static const int32_t s_IOR_%s_%s = %d;'
                 %(T, str.upper(sidl.method_id(m)), n))
        n = n + 1
    r.append('static const int32_t s_IOR_%s_MAX = %d;'%(T, n))
    r.append('''
static VAR_UNUSED struct {t}__method_desc{{
  const char* name;
  sidl_bool   is_static;
  long        est_interval;
  int         pre_complexity;
  int         post_complexity;
  double      meth_exec_time;
  double      pre_exec_time;
  double      post_exec_time;
}} s_ior_{t}_method[] = {{
'''.format(t=iorname))
    for m in cls.get_methods():
        r.append('{"%s", 1, 0, 0, 0, 0.0, 0.0, 0.0},'%sidl.method_id(m))

    r.append('};')
    r.append('')
    r.append('/* static structure options */')
    r.append('static const int32_t s_SEPV_%s_BASE      = 0;'%T)
    r.append('static const int32_t s_SEPV_%s_CONTRACTS = 1;'%T)
    r.append('static const int32_t s_SEPV_%s_HOOKS     = 2;'%T)
    r.append('')
    r.append('')
    return r

def generateMethodExecs(cls, iorname):
    """
    Generate the RStub source for a sidl class .  The source
    file begins with a banner and include files followed by a declaration
    of static methods and (for a class) external methods expected in the
    skeleton file.  For classes, the source file then defines a number
    of functions (cast, delete, initialize EPVs, new, init, and fini).
    """
    r = []

    # String my_symbolName = IOR.getSymbolName(id);
    # Collection c = new LinkedList(cls.getMethods(true));
    # String self = C.getObjectName(id) + " self";
    #  
    # //Adding cast as a possiblity for exec.  We can use if for addrefing connect!
    # c.add(IOR.getBuiltinMethod(IOR.CAST, cls.getSymbolID(), d_context, false));
    #  
    # //Do the same for the set_hooks builtin
    # if(d_context.getConfig().generateHooks())
    #   c.add(IOR.getBuiltinMethod(IOR.HOOKS, cls.getSymbolID(), d_context, false));
    
    for m in cls.all_methods:
        needExecErr = False
        if not sidl.is_static(m):
            numSerialized = 0  # number of args + return value + exceptions
            name = sidl.long_method_name(m)
            returnType = sidl.method_type_void(m)
            r.append("static void");
            r.append('%s_%s__exec('%(iorname,name))
            r.append("        struct %s__object* self,"%iorname);
            r.append("        struct sidl_rmi_Call__object* inArgs,");
            r.append("        struct sidl_rmi_Return__object* outArgs,");
            r.append("        struct sidl_BaseInterface__object** ex) {");
            r.append("}");

  #         d_writer.tab();
  #         List args = m.getArgumentList();
  #         comment("stack space for arguments");
  #         numSerialized = args.size();
  #         boolean hasReturn = returnType.getType() != Type.VOID;
  #         for (Iterator a = args.iterator(); a.hasNext(); ) { 
  #           Argument arg = (Argument) a.next();
  #           RMI.declareStackArgs(d_writer,arg, d_context);
  #         }
          
  #         if (hasReturn) {
  #           ++numSerialized;
  #           RMI.declareStackReturn(d_writer, returnType,m.isReturnCopy(),
  #                                  d_context);
  #         }
  #         r.append("sidl_BaseInterface _throwaway_exception = " + C.NULL 
  #                          + ";");
  #         r.append("sidl_BaseInterface _ex3   = " + C.NULL 
  #                          + ";");
  #         //A place to put exceptions that are thrown during serialization
  #         r.append("sidl_BaseException _SIDLex = " + C.NULL 
  #                          + ";");
        
  #         comment("unpack in and inout argments");
  #         for (Iterator a = args.iterator(); a.hasNext(); ) { 
  #           Argument arg = (Argument) a.next();
  #           if(arg.getMode() == Argument.IN || 
  #              arg.getMode() == Argument.INOUT) {
  #             RMI.unpackArg(d_writer, d_context, 
  #                           cls, "sidl_rmi_Call", "inArgs", arg, true);
  #           }
  #         }
  #         r.append();
        
  #         comment("make the call");
  #         if ( returnType.getType() != Type.VOID ) { 
  #           d_writer.print( "_retval = ");
  #         }
  #         if (m.isStatic()) {
  #           d_writer.print("(" 
  #                          + IOR.getStaticEPVVariable(id, IOR.EPV_STATIC, IOR.SET_PUBLIC) 
  #                          + "." + IOR.getVectorEntry(m.getLongMethodName()) + ")");
  #         }
  #         else {
  #           d_writer.print("(self->" + IOR.getEPVVar(IOR.PUBLIC_EPV) + "->" 
  #                          + IOR.getVectorEntry(m.getLongMethodName()) + ")");
  #         }
  #         r.append("(");
  #         d_writer.tab();
  #         IOR.generateArgumentList(d_writer, d_context,
  #                                  self, false, id, m, false, false, false,
  #                                "_ex", false, false, false, true );
  #         r.append(");  SIDL_CHECK(*_ex);");
  #         d_writer.backTab();
  #         r.append();
          
  #         comment("pack return value");
  #         if ( returnType.getType() != Type.VOID ) { 
  #           RMI.packType(d_writer, d_context,
  #                        "sidl_rmi_Return","outArgs", returnType, 
  #                        "_retval","_retval", Argument.IN, m.isReturnCopy(), 
  #                        false, true);
  #         }
       
  #         comment("pack out and inout argments");
  #         for (Iterator a = args.iterator(); a.hasNext(); ) { 
  #           Argument arg = (Argument) a.next();
  #           if (arg.getMode() == Argument.OUT || 
  #               arg.getMode() == Argument.INOUT) {
  #             RMI.packArg(d_writer, d_context,
  #                         "sidl_rmi_Return", "outArgs", arg, true);
  #           }
  #         }
       
  #         comment("clean-up dangling references");
       
  #         for (Iterator a = args.iterator(); a.hasNext(); ) { 
  #           Argument arg = (Argument) a.next();
  #           String mod = "";
  #           if(arg.getMode() != Argument.IN) {
  #             mod = "*";
  #           }
  #           if (arg.getType().getDetailedType() == Type.STRING) {
  #             r.append("if("+mod+arg.getFormalName()
  #                              +") {free("+mod+arg.getFormalName()+");}");
  #           } else if (arg.getType().getDetailedType() == Type.CLASS ||
  #                      arg.getType().getDetailedType() == Type.INTERFACE) {
            
  #             if(arg.getMode() == Argument.IN) {
  #               if(!arg.isCopy()) {
  #                 r.append("if("+arg.getFormalName()+
  #                                  "_str) {free("+arg.getFormalName()+"_str);}");
  #               }
  #               r.append("if("+mod+arg.getFormalName()+") {");
  #               d_writer.tab();
  #               r.append("sidl_BaseInterface_deleteRef((sidl_BaseInterface)"
  #                                +mod+arg.getFormalName()+", &_ex3); EXEC_CHECK(_ex3);");
  #               needExecErr = true;
  #               d_writer.backTab();
  #               r.append("}"); 
       
  #             } else { //Mode is OUT or INOUT, transfer the reference
  #               if(arg.isCopy()) {
  #                 r.append("if("+mod+arg.getFormalName()+") {");
  #                 d_writer.tab();
  #                 r.append("sidl_BaseInterface_deleteRef((sidl_BaseInterface)"
  #                                  +mod+arg.getFormalName()+", &_ex3); EXEC_CHECK(_ex3);");
  #                 needExecErr = true;
  #               } else {
  #                 r.append("if("+mod+arg.getFormalName()+
  #                                  " && sidl_BaseInterface__isRemote("+
  #                                  "(sidl_BaseInterface)"+mod+arg.getFormalName()+", &_throwaway_exception)) {");
                
  #                 d_writer.tab();
  #                 r.append("(*((sidl_BaseInterface)"+mod+arg.getFormalName()+
  #                                  ")->d_epv->f__raddRef)(((sidl_BaseInterface)"
  #                                  +mod+arg.getFormalName()+")->d_object, &_ex3); EXEC_CHECK(_ex3);");
  #                 needExecErr = true;
                
  #                 r.append("sidl_BaseInterface_deleteRef((sidl_BaseInterface)"+
  #                                  mod+arg.getFormalName()+", &_ex3); EXEC_CHECK(_ex3);");
  #               }
  #             d_writer.backTab();
  #             r.append("}"); 
       
  #             }
  #           } else if(arg.getType().getDetailedType() == Type.ARRAY) {
  #             r.append("sidl__array_deleteRef((struct sidl__array*)"+
  #                              mod+arg.getFormalName()+");");
  #           }
       
       
  #         }
       
  #         //The caller must ignore any (in)out parameters if an exception occured.
  #         //We get to EXIT whether we see an exception or not.
  #         //We then attempt to clean up.  Any exceptions in cleanup jump to EXEC_ERR.
  #         //EXEC_ERR then throws one or zero exceptions, giving priority to _ex.
  #         //If that fails, throw it back to the caller (probably an ORB).
  #         //DeleteRef the new exception, it should be duplicatable.
       
  #         // retval
  #         r.append("EXIT:");      
  #         if(returnType.getDetailedType() == Type.STRING) {
  #           r.append("if(_retval) {");
  #           d_writer.tab();
  #           r.append("free(_retval);");
  #           d_writer.backTab();
  #           r.append("}");
  #         } else if (returnType.getDetailedType() == Type.CLASS ||
  #                    returnType.getDetailedType() == Type.INTERFACE){
  #           if(m.isReturnCopy()) {
  #             r.append("if(_retval) {");
  #             d_writer.tab();
  #             r.append("sidl_BaseInterface_deleteRef((sidl_BaseInterface)"+
  #                              "_retval, &_ex3); EXEC_CHECK(_ex3);");
  #             needExecErr = true;
  #             d_writer.backTab();
  #             r.append("}");
  #           } else {
  #             r.append("if(_retval && sidl_BaseInterface__isRemote("+
  #                              "(sidl_BaseInterface)_retval, &_throwaway_exception)) {");
  #             d_writer.tab();
  #             r.append("(*((sidl_BaseInterface)_retval)->d_epv->f__raddRef)(((sidl_BaseInterface)_retval)->d_object, &_ex3); EXEC_CHECK(_ex3);");
            
  #             needExecErr = true;
  #             r.append("sidl_BaseInterface_deleteRef((sidl_BaseInterface)"+
  #                              "_retval, &_ex3); EXEC_CHECK(_ex3);");
  #             d_writer.backTab();
  #             r.append("}");
  #           }
  #         } else if(returnType.getDetailedType() == Type.ARRAY) {
  #           r.append("sidl__array_deleteRef((struct sidl__array*)_retval);");
  #         }
       
       
  #         // exec_err
  #         if (needExecErr) {
  #           r.append("EXEC_ERR:");
  #         }
          
  #         r.append("if(*_ex) { ");
  #         d_writer.tab();
  #         r.append("_SIDLex = sidl_BaseException__cast(*_ex,&_throwaway_exception);");
  #         r.append("sidl_rmi_Return_throwException(outArgs, _SIDLex, &_throwaway_exception); ");
       
  #         r.append("if(_throwaway_exception) {");
  #         d_writer.tab();
  #         comment("Throwing failed, throw _ex up the stack then.");
  #         r.append("sidl_BaseInterface_deleteRef(_throwaway_exception, &_throwaway_exception);");
  #         r.append("return;");
  #         d_writer.backTab();
  #         r.append("}");
       
  #         r.append("sidl_BaseException_deleteRef(_SIDLex, &_throwaway_exception);");
  #         r.append("sidl_BaseInterface_deleteRef(*_ex, &_throwaway_exception);");
  #         r.append("*_ex = NULL;");
  #         r.append("if(_ex3) { ");
  #         d_writer.tab();
  #         r.append("sidl_BaseInterface_deleteRef(_ex3, &_throwaway_exception);");
  #         r.append("_ex3 = NULL;");
  #         d_writer.backTab();
  #         r.append("}");
  #         d_writer.backTab();
  #         r.append("} else if(_ex3) {");
  #         d_writer.tab();
  #         r.append("_SIDLex = sidl_BaseException__cast(_ex3,&_throwaway_exception);");
  #         r.append("sidl_rmi_Return_throwException(outArgs, _SIDLex, &_throwaway_exception); ");
       
  #         r.append("if(_throwaway_exception) {");
  #         d_writer.tab();
  #         comment("Throwing failed throw _ex3 up the stack then.");
  #         r.append("sidl_BaseInterface_deleteRef(_throwaway_exception, &_throwaway_exception);");
  #         r.append("return;");
  #         d_writer.backTab();
  #         r.append("}");
       
  #         r.append("sidl_BaseException_deleteRef(_SIDLex, &_throwaway_exception);");
  #         r.append("sidl_BaseInterface_deleteRef(_ex3, &_throwaway_exception);");
  #         r.append("_ex3 = NULL;");
  #         d_writer.backTab();
  #         r.append("}");
  #         d_writer.backTab();
       
  #         r.append("}");
  #         r.append();
  #       }
  #     }
  #   }
    return '\n'.join(r)


def EPVinits(cls):
    r = []
    fixEPVs(r, cls, 0, is_new=True)
    return '\n'.join(r)
                 
                     
def fixEPVs(r, cls, level, is_new):
    """
    Recursively modify the EPVs in parent classes and set up interface
    pointers. Nothing is done if the class argument is null. The flag is_new
    determines whether the EPVs are being set to a newly defined EPV or to a
    previously saved EPV.
    """
    if not cls: return

    parent = cls.get_parent()
    fixEPVs(r, parent, level + 1, is_new)

    # Update the EPVs for all of the new interfaces in this particular class.
    _self    = 's%d' %level
    epvType  = 'my_' if is_new else 'par_'
    prefix   = '&' if is_new else ''
    ifce    = sorted(cls.get_unique_interfaces())
    epv     = 'epv'
    #width   = Utilities.getWidth(ifce) + epv.length() + 3;

    for i in ifce:
        name        = qual_id_low(i, '_')
        r.append('  %s->d_%s.d_%s = %ss_%s%s__%s;' %(_self, name, epv, prefix, epvType, epv, name))

    name_low = qual_cls_low(cls)
    name = qual_cls(cls)
    
    # Modify the class entry point vector.
    setContractsEPV = generateContractChecks(cls) and level == 0 and is_new
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
        r.append('      sidl_BaseInterface tae;');
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

    if generateBaseEPVAttr(cls):
        r.append('  %s->d_%s = %ss_%s%s__%s;'%(_self, epv, prefix, epvType, epv, name_low))

    r.append('')
                

def EPVfini(cls):
    r = []
    # Dump statistics (if enforcing contracts).
    if generateContractChecks(cls):
      r.append('  if (sidl_Enforcer_areEnforcing()) {')
      r.append('    (*(s0->d_epv->f__dump_stats))(s0, "", "FINI",_ex); SIDL_CHECK(*_ex);')
      r.append('  }');
      r.append('');

    # Call the user-defined destructor for this class.
    r.append('  (*(s0->d_epv->f__dtor))(s0,_ex); SIDL_CHECK(*_ex);')
 
    # If there is a parent class, then reset all parent pointers and call the
    # parent destructor.
    parent = cls.get_parent()
    if parent:
        r.append('')
        fixEPVs(r, parent, 1, is_new=False)
        r.append('  %s__fini(s1, _ex); SIDL_CHECK(*_ex);'%qual_cls(parent) )

    return '\n'.join(r)


def resolveRenamedMethods(ext, renames):
    
    # Check is current ext has renamed, if so add them
    renamed = ext.get_newmethods()
    for newM in renamed:
      oldM = ext.get_renamed_method()
      renames[oldM+"."+sidl.long_method_name(oldM)] = newM
    
    # Check if parent has any methods that are renamed, if so, add to map 
      for parent in ext.get_parents():
          for curM in parent.get_methods():
              # If the method was renamed in a child
              curM_name = sidl.long_method_name(curM)
              if (ext.getFullName() + "."+ curM_name) in renames:
                  # and we haven't already added this parents's version
                  k = parent.getFullName() + "." + curM_name
                  if k not in renames.containsKey():
                      # add this parents version
                      renames[k] = ext.lookup_method_by_long_name(sidl.long_method_name(curM), True)
            
          # After adding the methods from this parent, recursively call the parent
          resolveRenamedMethods(parent, renames)


@accepts(object)
def generateHookMethods(ext):
    """
    Return TRUE if hook methods are to be generated; FALSE otherwise.
    
    Assumptions:
    1) Assumptions in generateHookEPVs() apply.
    2) Hook methods are only generated if configuration indicates
       their generation is required.
    """
    return generateHookEPVs(ext) and gen_hooks

   
@accepts(object)
def generateHookEPVs(ext):
   """
   Return TRUE if the hooks-related EPVs are supposed to be generated.  
   
   Assumption:  Only non-SIDL interfaces and classes are to include 
   the hook EPVs.  Exceptions are _not_ to be included.
   """
   s_id = ext.get_scoped_id()
   return (not isSIDLSymbol(s_id)
           and not isSIDLXSymbol(s_id) 
           and not isException(ext))

@accepts(object)
def generateBaseEPVAttr(ext):
    """
    Return TRUE if the base EPV attribute needs to be supported; FALSE 
    otherwise.
    """   
    return generateHookMethods(ext) and \
        generateContractChecks(ext)
   

@accepts(object)
def generateContractChecks(ext):
    """
    Return TRUE if contract checks are supposed to be generated.
    
    Assumptions:
    1) Assumptions in generateContractEPVs() apply.
    2) Checks are only generated if the class has its own or
       inherited contract clauses.
    3) Checks are only generated if the configuration indicates
       their generation is required.
    """     
    return generateContractEPVs(ext) \
        and class_contracts(ext) \
        and gen_contracts
   

@accepts(object)
def class_contracts(cls):
    """
    Return TRUE if the class has any invariants or any methods define
    contracts
    """
    has_contracts = []

    def evaluate(_symtab, ext, _sid):
        if sidl.ext_invariants(ext):
            has_contracts.append(True)
            return

        for m in sidl.ext_methods(ext):
            if sidl.method_requires(m) or sidl.method_ensures(m):
                has_contracts.append(True)
                return

    if not has_contracts:
        visit_hierarchy(cls.data, evaluate, cls.symbol_table, [])

    return has_contracts == [True]

@accepts(object)
def generateContractEPVs(ext):
    """
    Return TRUE if the contract-related EPVs are supposed to be generated.  

    Assumptions:  
    1) Contract-related EPVs are only generated for concrete classes.
    2) Contract-related EPVs are not generated for SIDL classes.
    3) Contract-related EPVs are not generated for exceptions.
    """   
    return (not ext.is_abstract) and (not ext.is_interface()) \
        and generateContractBuiltins(ext)

@accepts(tuple)
def isSIDLSymbol(scoped_id):
    """
    Return TRUE if the Symbol ID corresponds to a SIDL symbol; FALSE 
    otherwise.
    """
    return sidl.scoped_id_modules(scoped_id)[0] == 'sidl'
   

@accepts(tuple)
def isSIDLXSymbol(scoped_id):
    """
    Return TRUE if the Symbol ID corresponds to a SIDLX symbol; FALSE 
    otherwise.
    """
    return sidl.scoped_id_modules(scoped_id)[0] == 'sidlx'

@accepts(object)
def generateContractBuiltins(ext):
    """
    Return TRUE if the contract-related built-in methods are to be
    generated.
    Assumptions:  
    1) Contract-related EPVs are not generated for SIDL interfaces or classes.
    2) Contract-related EPVs are not generated for exceptions.
    """
    s_id = ext.get_scoped_id()
    return ((not isSIDLSymbol(s_id))
            and (not isSIDLXSymbol(s_id))
            and (not isException(ext)))
   
   

@accepts(object)
def isException(ext):
   """
   Return <code>true</code> if and only if the extendable is
   a class that is the base exception class, is an interface that is
   the base exception interface, or it has the base exception class or 
   interface in its type ancestry.
   """
   base_ex = (sidl.scoped_id, ('sidl',), 'BaseException', '')
   sid = ext.get_id()
   return sid == base_ex or ext.has_parent_interface(base_ex)
