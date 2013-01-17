#!/usr/bin/env python
# -*- python -*-
## @package babel
#
# This file contains functions that implement backend-independent
# behavior of the Babel IOR.
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
# This file is part of BRAID ( http://compose-hpc.sourceforge.net/ ).
# Please read the COPYRIGHT file for Our Notice and for the BSD License.
#
# </pre>
#

import ir, sidl
from patmat import *
from sidl_symbols import SymbolTable
from string import Template

sidl_array_regex = re.compile('^sidl_((\w+)_)?_array$')

def strip_common(prefix, a):
    """
    \return \c a with the common prefix of \c prefix removed
    """
    while (len(prefix) and
           len(a) and
           prefix[0] == a[0]):

        a = a[1:]
        prefix = prefix[1:]
    return a

def qual_id(scoped_id, sep='_'):
    """
    Return the qualified name of a ScopedId in the form "prefix1.prefix2.name"+ext.
    \arg scoped_id the identifier
    \arg sep the separation character to use (default="_")
    """
    _, prefix, name, ext = scoped_id
    return sep.join(list(prefix)+[name])+ext

def qual_name(symbol_table, name, sep='_'):
    """
    Return the qualified name of an identifier in the form "prefix1.prefix2.name".
    \arg symbol_table     the \c SymbolTable
    \arg name             the identifier
    \arg sep              the separation character to use (default="_")
    """
    return sep.join(symbol_table.prefix+[name])


def object_type(package, name):
    """
    \return the SIDL node for the type of a Babel object 'name'
    \param name    the name of the object
    \param package the list of IDs making up the package
    """
    if isinstance(name, tuple):
        name = name[1]
    return sidl.Scoped_id(package, name, '')

def exception_type():
    """
    \return the SIDL node for the Babel exception type
    """
    return object_type(['sidl'], 'BaseInterface')

def ir_object_type(package, name):
    """
    \return the IR node for the type of a Babel object 'name'
    \param name    the name of the object
    \param package the list of IDs making up the package
    """
    return ir.Pointer_type(ir.Struct(ir.Scoped_id(package, name, '__object'), [], ''))

def ir_exception_type():
    """
    \return the IR node for the Babel exception type
    """
    return ir_object_type(['sidl'], 'BaseInterface')

def ir_baseinterface_type():
    """
    \return the IR node for the Babel exception type
    """
    return ir_object_type(['sidl'], 'BaseInterface')

def builtin((name, args)):
    return sidl.Method(sidl.void, sidl.Method_name(name, ''), [], args,
                       [], [], [], [], 'builtin method')

builtins = map(builtin,
               [('_ctor', []),
                ('_ctor2', [(sidl.arg, [], sidl.in_, ir.void_ptr, 'private_data')]),
                ('_dtor', []),
                ('_load', [])])

builtin_method_names = [
    "_cast",                    # the CAST method
    "_delete",			# the DELETE method
    "_exec",			# the reflexive EXEC method
    "_getURL",			# get's the object's URL (for RMI)
    "_raddRef",			# Remote addRef, Internal Babel
    "_isRemote",		# TRUE if this object is Remote
    "_set_hooks",		# the HOOKS method
    "_set_contracts",		# the Contract CONTRACTS method
    "_dump_stats",		# the DUMP_STATS method
    "_ctor",			# the CONSTRUCTOR method
    "_ctor2",			# the CONSTRUCTOR2 method
    "_dtor",			# the DESTRUCTOR method
    "_load"			# the LOAD method
  ]

rmi_related = [
    "_exec",
    "_getURL",
    "_raddRef",
    "_isRemote"
  ]


def argname((_arg, _attr, _mode, _type, Id)):
    return Id

def vcall(name, args, ci):
    """
    \return the IR for a non-static Babel virtual method call
    """
    try:
        cdecl = ci.epv.find_method(name)
        epv_type = ci.epv.get_type()
        epv = ir.Get_struct_item(ci.obj,
                                 ir.Deref(args[0]),
                                 ir.Struct_item(epv_type, 'd_epv'))
    except:
        if False:# FIXME no_contracts and no_hooks:
            return ir.Call('_'.join(ci.co.qualified_name+[name]), args)
        else:
            cdecl = ci.epv.find_static_method(name)
            epv_type = ci.epv.get_sepv_type()
            epv = '_getSEPV()'

    # this is part of an ugly hack to make sure that self is
    # dereferenced as self->d_object (by setting attr of self to the
    # unused value of 'pure')
    if ci.co.is_interface() and args:
        _, attrs, type_, id_, arguments, doc = cdecl
        _, attrs0, mode0, type0, name0 = arguments[0]
        arguments = [ir.Arg([ir.pure], mode0, type0, name0)]+arguments[1:]
        cdecl = ir.Fn_decl(attrs, type_, id_, arguments, doc)
            
    return ir.Call(ir.Deref(ir.Get_struct_item(
                epv_type,
                ir.Deref(epv),
                ir.Struct_item(ir.Pointer_type(cdecl), 'f_'+name))), args)

def drop_rarray_ext_args(args):
    """
    Now here gets funny: Since R-arrays are wrapped inside
    SIDL-Arrays in the IOR, convention says that we should remove all
    redundant arguments that can be derived from the SIDL-array's
    metadata.

    \bug{does not yet deal with nested expressions.}
    """
    names = set()
    for (arg, attrs, mode, typ, name) in args:
        if typ[0] == sidl.rarray:
            names.update(typ[3])

    return filter(lambda a: a[4] not in names, args)

# def struct_ior_names((struct, name, items, docstring)):
#     """
#     Append '__data' to a struct's name and all nested structs' names.
#     """
#     def f((item, typ, name)):
#         if typ[0] == ir.struct:
#             return item, struct_ior_names(typ), name
#         else: return item, typ, name
#
#     return struct, name+'__data', map(f, items), docstring


def notnone(fn):
    def wrapped(*args, **kw_args):
        r = fn(*args, **kw_args)
        if r == None:
            print args
            print '---->', r
            raise Exception("lower_ir() output failed sanity check")
        return r

    return wrapped

@notnone
@matcher(globals(), debug=False)
def lower_ir(symbol_table, sidl_term, header=None,
             struct_suffix='__data',
             enum_suffix='__enum',
             lower_scoped_ids=True,
             qualify_names=True,
             qualify_enums=True):
    """
    FIXME!! can we merge this with convert_arg??
    lower SIDL types into IR

    The idea is that no Chapel-specific code is in this function. It
    should provide a generic translation from SIDL -> IR.

    @param lower_scoped_ids  This is a broken design, but right now the
                             Chapel code generator accepts some sidl
                             node types such as array, rarray, and
                             class.  If False, then these types will
                             not be lowered.

    @param qualify_names     If \c True, enum values will get prefixed
                             with the full qualified name of the enum.
    """
    def low(sidl_term):
        return lower_ir(symbol_table, sidl_term, header,
                        struct_suffix, enum_suffix,
                        lower_scoped_ids,
                        qualify_names, qualify_enums)

    # print 'low(',sidl_term, ')'

    with match(sidl_term):
        if (sidl.arg, Attrs, Mode, (sidl.scoped_id, _, _, _), Name):
            lowtype = low(sidl_term[3])
            if lowtype[0] == ir.struct and Mode == ir.in_:
                # struct arguments are passed as pointer, regardless of mode
                # unless they are a return value
                lowtype = ir.Pointer_type(lowtype)
            return (ir.arg, Attrs, Mode, lowtype, Name)

        elif (sidl.arg, Attrs, Mode, Typ, Name):
            return (ir.arg, Attrs, Mode, low(Typ), Name)

        elif (sidl.scoped_id, Prefix, Name, Ext):
            return low(symbol_table[sidl_term][1])

        elif (sidl.void):                        return ir.pt_void
        elif (ir.void_ptr):                      return ir.void_ptr
        elif (sidl.primitive_type, sidl.opaque): return ir.Pointer_type(ir.pt_void)
        elif (sidl.primitive_type, sidl.string): return sidl_term #ir.const_str
        elif (sidl.primitive_type, Type):        return ir.Primitive_type(Type)

        elif (sidl.enum, Name, Enumerators, DocComment):
            if qualify_enums:
                es = lower_ir(SymbolTable(symbol_table,
                                          symbol_table.prefix+[Name]),
                              Enumerators, header, struct_suffix,
                              lower_scoped_ids, qualify_names)
                return ir.Enum(qual_name(symbol_table, sidl_term[1])+enum_suffix, es, DocComment)
            else:
                return ir.Enum(sidl_term[1], low(Enumerators), DocComment)

        elif (sidl.enumerator, Name):
            if qualify_enums: return ir.Enumerator(qual_name(symbol_table, Name))
            else:             return sidl_term

        elif (sidl.enumerator_value, Name, Val):
            if qualify_enums: return ir.Enumerator_value(qual_name(symbol_table, Name), Val)
            else:             return sidl_term


        elif (sidl.struct, (sidl.scoped_id, Prefix, Name, Ext), Items, DocComment):
            # a nested Struct
            return ir.Struct(qual_id(sidl_term[1])+struct_suffix, low(Items), '')

        elif (sidl.struct, Name, Items, DocComment):
            return ir.Struct(qual_name(symbol_table, Name)+struct_suffix, low(Items), '')

        elif (sidl.struct_item, Type, Name):
            if Type[0] == sidl.scoped_id:
                t = symbol_table[Type][1]
                if t[0] == sidl.class_ or t[0] == sidl.interface:
                    t = ir_object_type(t[1][1],t[1][2])
                elif t[0] == sidl.struct or t[0] == sidl.enum:
                    return (ir.struct_item, low(t), Name)
                return (ir.struct_item, t, Name)
            return (ir.struct_item, low(Type), Name)

        # elif (sidl.rarray, Scalar_type, Dimension, Extents):
        #     # Future optimization:
        #     # Direct-call version (r-array IOR)
        #     # return ir.Pointer_type(lower_type_ir(symbol_table, Scalar_type)) # FIXME
        #     # SIDL IOR version
        #     return ir.Typedef_type('sidl_%s__array'%Scalar_type[1])

        elif (sidl.rarray, Scalar_type, Dimension, Extents):
            return ir.Rarray(low(Scalar_type), Dimension, Extents)

        elif (sidl.array, [], [], []):
            #if not lower_scoped_ids: return sidl_term
            #return ir.Pointer_type(ir.pt_void)
            return ir.Pointer_type(ir.Struct('sidl__array', [], ''))

        elif (sidl.array, Scalar_type, Dimension, Orientation):
            #if not lower_scoped_ids: return sidl_term
            if Scalar_type[0] == ir.scoped_id:
                # FIXME: this is oversimplified, it should actually be
                # the real class name, but we don't yet generate array
                # declarations for all classes
                t = 'BaseInterface'
            else:
                t = Scalar_type[1]
            if header:
                header.genh(ir.Import('sidl_'+t+'_IOR'))
            return ir.Pointer_type(ir.Struct('sidl_%s__array'%t, [], ''))

        elif (sidl.class_, ScopedId, _, _, _, _, _):
            if not lower_scoped_ids: return ScopedId
            else: return ir_object_type(ScopedId[1], ScopedId[2])

        elif (sidl.interface, ScopedId, _, _, _, _):
            if not lower_scoped_ids: return ScopedId
            return ir_object_type(ScopedId[1], ScopedId[2])

        elif (Terms):
            if (isinstance(Terms, list)):
                return map(low, Terms)
            else:
                raise Exception("lower_ir: Not implemented: " + str(sidl_term))
        else: raise Exception("match error")


def get_type_name((fn_decl, Attrs, Type, Name, Args, DocComment)):
    return ir.Pointer_type((fn_decl, Attrs, Type, Name, Args, DocComment)), Name


def is_fixed_rarray(rarray):
    '''
    \return True iff the extent expressions of \c rarray are constant.
    '''
    return reduce(lambda a, b: a and b, map(lambda n: n > 0, rarray_len(rarray)))

@matcher(globals())
def rarray_len(rarray):
    '''
    \return a tuple of all the constant extent expressions of \c rarray.
    non-constant extents will be returned as -1
    '''
    def fail(Exp):
        import codegen
        print ("**ERROR: The %s expression is not supported "
               %codegen.sidl_gen(Exp)+
               "for rarray extent expressions.")
        exit(1)

    def l(extent):
        with match(extent):
            if (ir.simple_int_infix_expr, Op, A, B):
                if A < 0 or B < 0:    return -1
                elif Op == ir.plus:   return A+B
                elif Op == ir.minus:  return A-B
                elif Op == ir.times:  return A*B
                elif Op == ir.divide: return A//B
                elif Op == ir.modulo: return A%B
                elif Op == ir.lshift: return A<<B
                elif Op == ir.rshift: return A>>B
                elif Op == ir.pow:    return A**B
                else: fail(extent)

            elif (ir.simple_int_prefix_expr, Op, A):
                if A < 0:            return -1
                elif Op == ir.minus: return -A
                else: fail(extent)

            elif (ir.simple_int_fn_eval, Id, A):
                fail(extent)
                import pdb; pdb.set_trace()
                if A < 0: return -1
                else: fail(extent)

            elif (ir.var_ref, Id):
                return -1

            else:
                if isinstance(extent, int):
                    return extent
                fail(extent)

    return map(l, ir.rarray_extents(rarray))


class EPV(object):
    """
    Babel entry point vector for virtual method calls.

    Also contains the SEPV, which is used for all static functions, as
    well as the pre- and post-epv for the hooks implementation.

    """
    def __init__(self, class_object):
        self.methods = []
        self.static_methods = []
        # hooks
        self.pre_methods = []
        self.post_methods = []
        self.static_pre_methods = []
        self.static_post_methods = []

        self.name = class_object.name
        self.symbol_table = class_object.symbol_table
        self.has_static_methods = class_object.has_static_methods
        self.finalized = False

    def add_method(self, method):
        """
        add another (SIDL) method to the vector
        """
        def to_fn_decl((_sidl_method, Type,
                        (Method_name, Name, Extension),
                        Attrs, Args, Except, From, Requires, Ensures, DocComment),
                       suffix=''):
            typ = lower_ir(self.symbol_table, Type)
            name = 'f_'+Name+Extension

            # discard the abstract/final attributes. Ir doesn't know them.
            attrs = set(Attrs)
            attrs.discard(sidl.abstract)
            attrs.discard(sidl.final)
            attrs = list(attrs)
            args = epv_args(attrs, Args, self.symbol_table, self.name)
            return ir.Fn_decl(attrs, typ, name+suffix, args, DocComment)

        if self.finalized:
            import pdb; pdb.set_trace()
            raise Exception()

        if member_chk(sidl.static, method[3]):
            self.static_methods.append(to_fn_decl(method))
            if member_chk(ir.hooks, method[3]):
                self.static_pre_methods.append(to_fn_decl(method, '_pre'))
                self.static_post_methods.append(to_fn_decl(method, '_post'))
        else:
            self.methods.append(to_fn_decl(method))
            if member_chk(ir.hooks, method[3]):
                self.pre_methods.append(to_fn_decl(method, '_pre'))
                self.post_methods.append(to_fn_decl(method, '_post'))
        return self

    def find_method(self, method):
        """
        Perform a linear search through the list of methods and return
        the first with a matching name.
        """
        for m in self.methods:
            fn_decl, attrs, typ, name, args, doc = m
            if name == 'f_'+method:
                return fn_decl, attrs, typ, name[2:], args, doc
        raise Exception()

    def find_static_method(self, method):
        """
        Perform a linear search through the list of static methods and return
        the first with a matching name.
        """
        for m in self.static_methods:
            fn_decl, attrs, typ, name, args, doc = m
            if name == 'f_'+method:
                return fn_decl, attrs, typ, name[2:], args, doc
        raise Exception()

    def get_ir(self):
        """
        return an s-expression of the EPV declaration
        """

        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__epv', '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname)
             for itype, iname in map(get_type_name, self.methods)],
                         'Entry Point Vector (EPV)')

    def get_sepv_ir(self):
        """
        return an s-expression of the SEPV declaration
        """

        if not self.has_static_methods:
            return ''

        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__sepv', '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname)
             for itype, iname in map(get_type_name, self.static_methods)],
                         'Static Entry Point Vector (SEPV)')

    nonempty = ir.Struct_item(ir.pt_char, 'd_not_empty')

    def get_pre_epv_ir(self):
        """
        return an s-expression of the pre_EPV declaration
        """
        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__pre_epv', '')
        elems = [ir.Struct_item(itype, iname)
                 for itype, iname in map(get_type_name, self.pre_methods)]
        if elems == []: elems = [self.nonempty]
        return ir.Struct(name, elems, 'Pre Hooks Entry Point Vector (pre_EPV)')

    def get_post_epv_ir(self):
        """
        return an s-expression of the post_EPV declaration
        """
        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__post_epv', '')
        elems = [ir.Struct_item(itype, iname)
                 for itype, iname in map(get_type_name, self.post_methods)]
        if elems == []: elems = [self.nonempty]
        return ir.Struct(name, elems, 'Pre Hooks Entry Point Vector (post_EPV)')

    def get_pre_sepv_ir(self):
        """
        return an s-expression of the pre_SEPV declaration
        """
        if not self.has_static_methods:
            return ''

        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__pre_sepv', '')

        if self.static_post_methods:
            entries = [ir.Struct_item(itype, iname)
                       for itype, iname in
                       map(get_type_name, self.static_pre_methods)]
        else:
            entries = [self.nonempty]
        return ir.Struct(name, entries, 'Pre Hooks Entry Point Vector (pre_EPV)')

    def get_post_sepv_ir(self):
        """
        return an s-expression of the post_SEPV declaration
        """
        if not self.has_static_methods:
            return ''

        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__post_sepv', '')

        if self.static_post_methods:
            entries = [ir.Struct_item(itype, iname)
                       for itype, iname in
                       map(get_type_name, self.static_post_methods)]
        else:
            entries = [self.nonempty]
        return ir.Struct(name, entries, 'Post Hooks Entry Point Vector (post_EPV)')


    def get_type(self):
        """
        return an s-expression of the EPV's (incomplete) type
        """
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__epv', '')
        return ir.Struct(name, [], 'Entry Point Vector (EPV)')

    def get_sepv_type(self):
        """
        return an s-expression of the SEPV's (incomplete) type
        """
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__sepv', '')
        return ir.Struct(name, [], 'Static Entry Point Vector (SEPV)')


def static_ior_args(args, symbol_table, class_name):
    """
    \return a SIDL -> Ir lowered version of
    [self]+args+(sidl_BaseInterface__object*)[*ex]
    """
    arg_self = [ir.Arg([], ir.in_,
                       ir_object_type(symbol_table.prefix, class_name),
                       'self')]
    arg_ex = [ir.Arg([], sidl.out, ir_baseinterface_type(), '_ex')]
    return arg_self+lower_ir(symbol_table, args, qualify_names=True)+arg_ex


def epv_args(attrs, args, symbol_table, class_name):
    """
    \return a SIDL -> Ir lowered version of [self]+args+[*ex]
    """
    if member_chk(sidl.static, attrs):
        arg_self = []
    else:
        arg_self = \
            [ir.Arg([], ir.in_,
                    ir_object_type(symbol_table.prefix, class_name),
                    'self')]
    arg_ex = \
        [ir.Arg([], ir.out, ir_baseinterface_type(), '_ex')]
    return arg_self+lower_ir(symbol_table, args, qualify_names=True)+arg_ex

def stub_args(attrs, args, symbol_table, class_name, extra_attrs):
    """
    \return a SIDL -> [*self]+args+[*ex]
    """
    if member_chk(sidl.static, attrs):
        arg_self = []
    else:
        arg_self = [
            ir.Arg(extra_attrs, sidl.in_,
                ir_object_type(symbol_table.prefix, class_name), 'self')]
    arg_ex = \
        [ir.Arg(extra_attrs, sidl.out, ir_exception_type(), '_ex')]
    return arg_self+args+arg_ex


def is_obj_type(symbol_table, typ):
    return typ[0] == sidl.scoped_id and (
        symbol_table[typ][1][0] == sidl.class_ or
        symbol_table[typ][1][0] == sidl.interface)

def is_struct_type(symbol_table, typ):
    return typ[0] == sidl.scoped_id and symbol_table[typ][1][0] == sidl.struct


def ior_type(symbol_table, t):
    """
    if \c t is a scoped_id return the IOR type of t.
    else return \c t.
    """
    if (t[0] == sidl.scoped_id and symbol_table[t][1][0] in [sidl.class_, sidl.interface]):
    #    return ir_object_type(*symbol_table.get_full_name(t[1]))
        return ir_object_type(t[1], t[2])

    else: return t


def externals(scopedid):
    return Template('''
#include "sidlOps.h"
#include "sidl.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_rmi_InstanceHandle_IOR.h"

// Avoid mixing Braid and Babel headers
#ifndef included_sidl_BaseException_h
#define included_sidl_BaseException_h
#endif
#include "sidl_Exception.h"

// Hold pointer to IOR functions.
static const struct ${ext}__external *_externals = NULL;

extern const struct ${ext}__external* ${ext}__externals(void);

// Lookup the symbol to get the IOR functions.
static const struct ${ext}__external* _loadIOR(void)

// Return pointer to internal IOR functions.
{
#ifdef SIDL_STATIC_LIBRARY
  _externals = ${ext}__externals();
#else
  _externals = (struct ${ext}__external*)sidl_dynamicLoadIOR(
    "${ext_dots}","${ext}__externals") ;
  sidl_checkIORVersion("${ext_dots}", _externals->d_ior_major_version,
    _externals->d_ior_minor_version, 2, 0);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

// Hold pointer to static entry point vector
static const struct ${ext}__sepv *_sepv = NULL;

// Return pointer to static functions.
#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getExternals()->getStaticEPV))()))

// Reset point to static functions.
#define _resetSEPV() (_sepv = (*(_getExternals()->getStaticEPV))())

''').substitute(ext=qual_id(scopedid), ext_dots=qual_id(scopedid, '.'))

def builtin_stub_functions(scopedid):
    '''
    builtins ususally included in the C-bases server stub
    '''
    return Template('''
/*
 * Constructor function for the class.
 */
#pragma weak ${ext}__create
${ext}
${ext}__create(sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(NULL,_ex);
}

/**
 * Wraps up the private data struct pointer (struct ${ext}__data) passed in rather than running the constructor.
 */
#pragma weak ${ext}__wrapObj
${ext}
${ext}__wrapObj(void* data, sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(data, _ex);
}

#ifdef WITH_RMI

static ${ext} ${ext}__remoteCreate(const char* url, sidl_BaseInterface
  *_ex);
/*
 * RMI constructor function for the class.
 */
#pragma weak ${ext}__createRemote
${ext}
${ext}__createRemote(const char* url, sidl_BaseInterface *_ex)
{
  return NULL;//${ext}__remoteCreate(url, _ex);
}

static struct ${ext}__object* ${ext}__remoteConnect(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {return 0;/*FIXME*/}
static struct ${ext}__object* ${ext}__IHConnect(struct
  sidl_rmi_InstanceHandle__object* instance, sidl_BaseInterface *_ex) {return NULL;/*FIXME*/}

/*
 * RMI connector function for the class.
 */
#pragma weak ${ext}__connect
${ext}
${ext}__connect(const char* url, sidl_BaseInterface *_ex)
{
  return NULL;//${ext}__remoteConnect(url, TRUE, _ex);
}

#endif /*WITH_RMI*/

/*
 * Method to enable/disable interface contract enforcement.
 */
#pragma weak ${ext}__set_contracts
void
${ext}__set_contracts(
  ${ext} self,
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex)
{
  (*self->d_epv->f__set_contracts)(
  self,
  enable, enfFilename, resetCounters, _ex);
}

/*
 * Method to dump interface contract enforcement statistics.
 */
#pragma weak ${ext}__dump_stats
void
${ext}__dump_stats(
  ${ext} self,
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex)
{
  (*self->d_epv->f__dump_stats)(
  self,
  filename, prefix, _ex);
}

/*
 * Cast method for interface and class type conversions.
 */
#pragma weak ${ext}__cast
${ext}
${ext}__cast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  ${ext} cast = NULL;

#ifdef WITH_RMI
  static int connect_loaded = 0;
  if (!connect_loaded) {
    connect_loaded = 1;
    sidl_rmi_ConnectRegistry_registerConnect("${ext_dots}", (
      void*)${ext}__IHConnect,_ex);SIDL_CHECK(*_ex);
  }
#endif /*WITH_RMI*/
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (${ext}) (*base->d_epv->f__cast)(
      base->d_object,
      "${ext_dots}", _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */
#pragma weak ${ext}__cast2
void*
${ext}__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface* _ex)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type, _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}




// FIXME: this does not belong here. It should be in stub.h
#pragma weak ${ext}__isRemote
sidl_bool
${ext}__isRemote(
  /* in */ ${ext} self,
  /* out */ sidl_BaseInterface *_ex)
{
  sidl_bool _result;
  _result = (*self->d_epv->f__isRemote)(
    self,
    _ex);
  return _result;
}

/*
 * TRUE if this object is remote, false if local
 */
#pragma weak ${ext}__isLocal
sidl_bool
${ext}__isLocal(
  /* in */ ${ext} self,
  /* out */ sidl_BaseInterface *_ex)
{
  return !${ext}__isRemote(self, _ex);
}

''').substitute(ext=qual_id(scopedid), ext_dots=qual_id(scopedid, '.'))

def build_function_call(ci, cdecl, static):
    '''
    Build an IR expression that consists of the EPV lookup for a
    possibly virtual function call using Babel IOR.
    '''
    #if final:
        # final : Final methods are the opposite of virtual. While
        # they may still be inherited by child classes, they
        # cannot be overridden.

        # static call
        #The problem with this is that e.g. C++ symbols usere different names
        #We should modify Babel to generate  __attribute__ ((weak, alias ("__entry")))
    #    callee = '_'.join(['impl']+symbol_table.prefix+[ci.epv.name,Name])
    if static:
        # static : Static methods are sometimes called "class
        # methods" because they are part of a class, but do not
        # depend on an object instance. In non-OO languages, this
        # means that the typical first argument of an instance is
        # removed. In OO languages, these are mapped directly to
        # an Java or C++ static method.
        epv_type = ci.epv.get_type()
        obj_type = ci.obj
        callee = ir.Get_struct_item(
            epv_type,
            ir.Deref(ir.Call('_getSEPV', [])),
            ir.Struct_item(ir.Pointer_type(cdecl), 'f_'+ir.fn_decl_id(cdecl)))

    else:
        # dynamic virtual method call
        epv_type = ci.epv.get_type()
        obj_type = ci.obj
        callee = ir.Deref(ir.Get_struct_item(
            epv_type,
            ir.Deref(ir.Get_struct_item(obj_type,
                                        ir.Deref('self'),
                                        ir.Struct_item(epv_type, 'd_epv'))),
            ir.Struct_item(ir.Pointer_type(cdecl), 'f_'+ir.fn_decl_id(cdecl))))

    return callee
