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
    return sep.join(prefix+[name])+ext

def babel_object_type(package, name):
    """
    \return the SIDL node for the type of a Babel object 'name'
    \param name    the name of the object
    \param package the list of IDs making up the package
    """
    if isinstance(name, tuple):
        name = name[1]
    return sidl.Scoped_id(package, name, '')

def babel_exception_type():
    """
    \return the SIDL node for the Babel exception type
    """
    return babel_object_type(['sidl'], 'BaseInterface')

def ir_babel_object_type(package, name):
    """
    \return the IR node for the type of a Babel object 'name'
    \param name    the name of the object
    \param package the list of IDs making up the package
    """
    return ir.Pointer_type(ir.Struct(babel_object_type(package, name+'__object'), [], ''))

def ir_babel_exception_type():
    """
    \return the IR node for the Babel exception type
    """
    return ir_babel_object_type(['sidl'], 'BaseInterface')

def ir_babel_baseinterface_type():
    """
    \return the IR node for the Babel exception type
    """
    return ir_babel_object_type(['sidl'], 'BaseInterface')

def builtin((name, args)):
    return sidl.Method(sidl.void, sidl.Method_name(name, ''), [], args,
                       [], [], [], [], 'builtin method')

builtins = map(builtin,
               [('_ctor', []), 
                ('_ctor2', [(sidl.arg, [], sidl.in_, ir.void_ptr, 'private_data')]),
                ('_dtor', []),
                ('_load', [])])


def argname((_arg, _attr, _mode, _type, Id)):
    return Id

def vcall(name, args, ci):
    """
    \return the IR for a non-static Babel virtual method call
    """
    epv = ci.epv.get_type()
    cdecl = ci.epv.find_method(name)
    # this is part of an ugly hack to make sure that self is
    # dereferenced as self->d_object (by setting attr of self to the
    # unused value of 'pure')
    if ci.co.is_interface() and args:
        _, attrs, type_, id_, arguments, doc = cdecl
        _, attrs0, mode0, type0, name0 = arguments[0]
        arguments = [ir.Arg([ir.pure], mode0, type0, name0)]+arguments[1:]
        cdecl = ir.Fn_decl(attrs, type_, id_, arguments, doc)
        
    return ir.Stmt(ir.Call(ir.Deref(ir.Get_struct_item(epv,
                ir.Deref(ir.Get_struct_item(ci.obj,
                                            ir.Deref('self'),
                                            ir.Struct_item(epv, 'd_epv'))),
                ir.Struct_item(ir.Pointer_type(cdecl), 'f_'+name))), args))

def drop_rarray_ext_args(args):
    """
    Now here it's becoming funny: Since R-arrays are wrapped inside
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

def struct_ior_names((struct, name, items, docstring)):
    """
    Append '__data' to a struct's name and all nested structs' names. 
    """
    def f((item, typ, name)):
        if typ[0] == ir.struct:
            return item, struct_ior_names(typ), name
        else: return item, typ, name

    return struct, name+'__data', map(f, items), docstring


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
def lower_ir(symbol_table, sidl_term, header=None, struct_suffix='__data', lower_scoped_ids=True):
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
    """
    def low(sidl_term):
        return lower_ir(symbol_table, sidl_term, header, struct_suffix, lower_scoped_ids)

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
        elif (sidl.primitive_type, sidl.string): return ir.const_str
        elif (sidl.primitive_type, sidl.bool):   return ir.Typedef_type('sidl_bool')
        elif (sidl.primitive_type, sidl.long):   return ir.Typedef_type('int64_t')
        elif (sidl.primitive_type, Type):        return ir.Primitive_type(Type)
        elif (sidl.enum, _, _, _):               return sidl_term # identical
        elif (sidl.enumerator, _):               return sidl_term
        elif (sidl.enumerator, _, _):            return sidl_term


        elif (sidl.struct, (sidl.scoped_id, Prefix, Name, Ext), Items, DocComment):
            # a nested Struct
            return ir.Struct(qual_id(sidl_term[1])+struct_suffix, low(Items), '')

        elif (sidl.struct, Name, Items, DocComment):
            qname = '_'.join(symbol_table.prefix+[Name])
            return ir.Struct(qname, low(Items), '')

        elif (sidl.struct_item, Type, Name):
            return (ir.struct_item, low(Type), Name)

        # elif (sidl.rarray, Scalar_type, Dimension, Extents):
        #     # Future optimization:
        #     # Direct-call version (r-array IOR)
        #     # return ir.Pointer_type(lower_type_ir(symbol_table, Scalar_type)) # FIXME
        #     # SIDL IOR version
        #     return ir.Typedef_type('sidl_%s__array'%Scalar_type[1])

        elif (sidl.rarray, Scalar_type, Dimension, Extents):
            if not lower_scoped_ids: return sidl_term
            # Rarray appearing inside of a struct
            return ir.Pointer_type(Scalar_type)

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
            else: return ir_babel_object_type(ScopedId[1], ScopedId[2])
        
        elif (sidl.interface, ScopedId, _, _, _, _):
            if not lower_scoped_ids: return ScopedId
            return ir_babel_object_type(ScopedId[1], ScopedId[2])
        
        elif (Terms):
            if (isinstance(Terms, list)):
                return map(low, Terms)
        else:
            raise Exception("lower_ir: Not implemented: " + str(sidl_term))


def get_type_name((fn_decl, Attrs, Type, Name, Args, DocComment)):
    return ir.Pointer_type((fn_decl, Attrs, Type, Name, Args, DocComment)), Name

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
            args = babel_epv_args(attrs, Args, self.symbol_table, self.name)
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
        import pdb; pdb.set_trace()
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

    def get_pre_epv_ir(self):
        """
        return an s-expression of the pre_EPV declaration
        """
        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__pre_epv', '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname)
             for itype, iname in map(get_type_name, self.pre_methods)],
                         'Pre Hooks Entry Point Vector (pre_EPV)')

    def get_post_epv_ir(self):
        """
        return an s-expression of the post_EPV declaration
        """
        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix, self.name+'__post_epv', '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname)
             for itype, iname in map(get_type_name, self.post_methods)],
                         'Pre Hooks Entry Point Vector (post_EPV)')

    nonempty = ir.Struct_item(ir.pt_char, 'd_not_empty')

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


def babel_static_ior_args(args, symbol_table, class_name):
    """
    \return a SIDL -> Ir lowered version of 
    [self]+args+(sidl_BaseInterface__object*)[*ex]
    """
    arg_self = [ir.Arg([], ir.in_, 
                       ir_babel_object_type(symbol_table.prefix, class_name),
                       'self')]
    arg_ex = [ir.Arg([], sidl.out, ir_babel_baseinterface_type(), '_ex')]
    return arg_self+lower_ir(symbol_table, args)+arg_ex


def babel_epv_args(attrs, args, symbol_table, class_name):
    """
    \return a SIDL -> Ir lowered version of [self]+args+[*ex]
    """
    if member_chk(sidl.static, attrs):
        arg_self = []
    else:
        arg_self = \
            [ir.Arg([], ir.in_, 
                    ir_babel_object_type(symbol_table.prefix, class_name),
                    'self')]
    arg_ex = \
        [ir.Arg([], ir.out, ir_babel_baseinterface_type(), '_ex')]
    return arg_self+lower_ir(symbol_table, args)+arg_ex

def babel_stub_args(attrs, args, symbol_table, class_name, extra_attrs):
    """
    \return a SIDL -> [*self]+args+[*ex]
    """
    if member_chk(sidl.static, attrs):
        arg_self = []
    else:
        arg_self = [
            ir.Arg(extra_attrs, sidl.in_, 
                ir_babel_object_type(symbol_table.prefix, class_name), 'self')]
    arg_ex = \
        [ir.Arg(extra_attrs, sidl.out, ir_babel_exception_type(), '_ex')]
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
    #    return ir_babel_object_type(*symbol_table.get_full_name(t[1]))
        return ir_babel_object_type(t[1], t[2])

    else: return t


def externals(scopedid):
    return '''
#include "sidlOps.h"

// Hold pointer to IOR functions.
static const struct {a}__external *_externals = NULL;

extern const struct {a}__external* {a}__externals(void);

// Lookup the symbol to get the IOR functions.
static const struct {a}__external* _loadIOR(void)

// Return pointer to internal IOR functions.
{{
#ifdef SIDL_STATIC_LIBRARY
  _externals = {a}__externals();
#else
  _externals = (struct {a}__external*)sidl_dynamicLoadIOR(
    "{b}","{a}__externals") ;
  sidl_checkIORVersion("{b}", _externals->d_ior_major_version, 
    _externals->d_ior_minor_version, 2, 0);
#endif
  return _externals;
}}

#define _getExternals() (_externals ? _externals : _loadIOR())

// Hold pointer to static entry point vector
static const struct {a}__sepv *_sepv = NULL;

// Return pointer to static functions.
#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getExternals()->getStaticEPV))()))

// Reset point to static functions.
#define _resetSEPV() (_sepv = (*(_getExternals()->getStaticEPV))())

'''.format(a=qual_id(scopedid), b=qual_id(scopedid, '.'))

