#!/usr/bin/env python
# -*- python -*-
## @package chapel
#
# Babel functionality for the Chapel PGAS language
# http://chapel.cray.com/
#
# Please report bugs to <adrian@llnl.gov>.
#
# \authors <pre>
#
# Copyright (c) 2011, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
#
# LLNL-CODE-473891.
# All rights refserved.
#
# This file is part of BRAID. For details, see
# http://compose-hpc.sourceforge.net/.
# Please read the COPYRIGHT file for Our Notice and
# for the BSD License.
#
# </pre>
#

import config, ir, os, re, sidl, types
from patmat import matcher, match, member, unify, expect, Variable, unzip
from codegen import (
    ClikeCodeGenerator, CCodeGenerator,
    SourceFile, CFile, Scope, generator, accepts,
    sep_by
)

chpl_data_var_template = '_babel_data_{arg_name}'
chpl_dom_var_template = '_babel_dom_{arg_name}'
chpl_local_var_template = '_babel_local_{arg_name}'

def drop(lst):
    """
    If \c lst is \c [] return \c None, else return \c lst[0] .
    """
    if lst: return lst[0]
    else:   return None

def babel_object_type(package, name):
    """
    \return the SIDL node for the type of a Babel object 'name'
    \param name    the name of the object
    \param package the list of IDs making up the package
    """
    if isinstance(name, tuple):
        name = name[1]
    elif isinstance(name, str): name = [name]
    return sidl.Scoped_id(package+name, "")

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
    if isinstance(name, tuple):
        name = name[1]
    elif isinstance(name, str): name = [name]
    return ir.Struct(babel_object_type(package+name, '_object'), [], '')

def ir_babel_exception_type():
    """
    \return the IR node for the Babel exception type
    """
    return ir.Pointer_type(ir_babel_object_type(['sidl'], 'BaseInterface'))

def argname((_arg, _attr, _mode, _type, Id)):
    return Id

@accepts(str, str)
def write_to(filename, string):
    """
    Create/Overwrite a file named \c filename with the contents of \c
    string.
    The file is written atomically.
    """
    tmp = '#'+filename+'#'
    f = open(tmp,'w')
    f.write(string)
    f.flush()
    os.fsync(f)
    f.close()
    os.rename(tmp, filename)


class Chapel:
    
    class ClassInfo:
        """
        Holder object for the code generation scopes and other data
        during the traversal of the SIDL tree.
        """
        def __init__(self, name, symbol_table):
            self.impl = ChapelFile()
            self.chpl_stub = ChapelFile(relative_indent=4)
            self.chpl_static_stub = ChapelFile(self.chpl_stub)            
            self.skel = CFile()
            self.epv = EPV(name, symbol_table)
            self.ior = CFile()
            self.obj = None

    def __init__(self, filename, sidl_sexpr, symbol_table, create_makefile, verbose):
        """
        Create a new chapel code generator
        \param filename        full path to the SIDL file
        \param sidl_sexpr      s-expression of the SIDL data
        \param symbol_table    symbol table for the SIDL data
        \param create_makefile if \c True, also generate a GNUmakefile
        \param verbose         if \c True, print extra messages
        """
        self.sidl_ast = sidl_sexpr
        self.symbol_table = symbol_table
        self.sidl_file = filename
        self.create_makefile = create_makefile
        self.verbose = verbose
        self.classes = []

    def generate_client(self):
        """
        Generate client code. Operates in two passes:
        \li create symbol table
        \li do the work
        """
        try:
            self.generate_client1(self.sidl_ast, None, self.symbol_table)
            if self.create_makefile:
                generate_client_makefile(self.sidl_file, self.classes)
        except:
            # Invoke the post-mortem debugger
            import pdb, sys
            print sys.exc_info()
            pdb.post_mortem()

    def generate_server(self):
        """
        Generate server code. Operates in two passes:
        \li create symbol table
        \li do the work
        """
        try:
            self.generate_server1(self.sidl_ast, None, self.symbol_table)
            self.generate_server_makefile(self.classes)

        except:
            # Invoke the post-mortem debugger
            import pdb, sys
            print sys.exc_info()
            pdb.post_mortem()


    @matcher(globals(), debug=False)
    def generate_client1(self, node, data, symbol_table):
        """
        CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT
        """
        def gen(node):         return self.generate_client1(node, data, symbol_table)
        def gen1(node, data1): return self.generate_client1(node, data1, symbol_table)

        if not symbol_table:
            raise Exception()

        with match(node):
            if (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                self.generate_client_method(symbol_table, node, data)

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                ci = self.ClassInfo(Name, symbol_table)
                ci.chpl_stub.cstub.genh(ir.Import(Name+'_IOR'))
                ci.chpl_stub.cstub.genh(ir.Import('sidlType'))
                ci.chpl_stub.cstub.genh(ir.Import('chpl_sidl_array'))
                ci.chpl_stub.cstub.genh(ir.Import('chpltypes'))
                
                self.gen_default_methods(symbol_table, Name, ci)

                # recurse to generate method code
                gen1(Methods, ci)

                # IOR
                self.generate_ior(ci)
                write_to(Name+'_IOR.h', ci.ior.dot_h(Name+'_IOR.h'))

                # Stub (in C)
                cstub = ci.chpl_stub.cstub
                cstub.genh_top(ir.Import(Name+'_IOR'))
                for code in cstub.optional:
                    cstub.new_global_def(code)
                    
                cstub.gen(ir.Import(Name+'_cStub'))
                
                # Stub Header
                write_to(Name+'_cStub.h', cstub.dot_h(Name+'_cStub.h'))
                # Stub C-file
                write_to(Name+'_cStub.c', cstub.dot_c())

                # Stub (in Chapel)
                qname = '_'.join(symbol_table.prefix+[Name])                
                # Chapel supports C structs via the _extern keyword,
                # but they must be typedef'ed in a header file that
                # must be passed to the chpl compiler.
                typedefs = CFile();
                typedefs._header = [
                    '// Package header (enums, etc...)',
                    '#include <stdint.h>',
                    '#include <%s.h>' % '_'.join(symbol_table.prefix),
                    '#include <%s_IOR.h>'%Name,
                    'typedef struct %s__object _%s__object;'%(qname, qname),
                    'typedef _%s__object* %s__object;'%(qname, qname),
                    '#ifndef _CHPL_SIDL_BASETYPES',
                    '#define _CHPL_SIDL_BASETYPES',
                    'typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;',
                    'typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;',
                    '%s__object %s__createObject(%s__object copy, sidl_BaseInterface__object* ex);'
                    %(qname, qname, qname),
                    '#endif'
                    ]
                write_to(Name+'_Stub.h', typedefs.dot_h(Name+'_Stub.h'))
                chpl_defs = ci.chpl_stub
                ci.chpl_stub = ChapelFile()
                ci.chpl_stub.new_def('use sidl;')
                ci.chpl_stub.new_def('_extern record %s__object {};'%qname)
                ci.chpl_stub.new_def('_extern proc %s__createObject('%qname+
                                     'd_data: int, '+
                                     'inout ex: sidl_BaseInterface__object)'+
                                     ': %s__object;'%qname)
                name = chpl_gen(Name)
                ci.chpl_stub.new_def(chpl_defs.get_decls())
                ci.chpl_stub.new_def('// All the static methods of class '+name)
                ci.chpl_stub.new_def('module %s_static {'%name)
                ci.chpl_stub.new_def(ci.chpl_static_stub.get_defs())
                ci.chpl_stub.new_def('}')
                ci.chpl_stub.new_def('class %s {'%name)
                chpl_class = ChapelScope(ci.chpl_stub)
                chpl_class.new_def('var self: %s__object;'%qname)
                body = [
                    '  var ex: sidl_BaseInterface__object;',
                    '  this.self = %s__createObject(0, ex);'%qname
                    ]
                chpl_gen(
                    (ir.fn_defn, [], ir.pt_void, 
                     Name, [], body, 'Constructor'), chpl_class)

                chpl_gen(
                    (ir.fn_defn, [], ir.pt_void, 
                     chpl_gen(Name),
                     [ir.Arg([], ir.in_, ir_babel_object_type([], qname), 'obj')],
                      ['  this.self = obj;'],
                      'Constructor for wrapping an existing object'), chpl_class)

                chpl_class.new_def(chpl_defs.get_defs())
                ci.chpl_stub.new_def(chpl_class)
                ci.chpl_stub.new_def('}')
                self.pkg_chpl_stub.new_def(ci.chpl_stub)

                # Makefile
                self.classes.append(Name)

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # do nothing for now / although the interface should
                # be used for correctness checks, we probably don't
                # need to emit any code for it, do we?
                pass

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_stub.gen(ir.Type_decl(node))
                self.pkg_enums.append(node)
                
            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel stub
                self.pkg_chpl_stub = ChapelFile()
                self.pkg_enums = []
                self.generate_client1(UserTypes, data, symbol_table[[Name]])
                write_to(Name+'.chpl', str(self.pkg_chpl_stub))

                pkg_h = CFile()
                for enum in self.pkg_enums:
                    pkg_h.gen(ir.Type_decl(enum))
                write_to(Name+'.h', pkg_h.dot_h(Name+'.h'))

            elif (sidl.user_type, Attrs, Cipse):
                gen(Cipse)

            elif (sidl.file, Requires, Imports, UserTypes):
                gen(UserTypes)

            elif A:
                if (isinstance(A, list)):
                    for defn in A:
                        gen(defn)
                else:
                    raise Exception("NOT HANDLED:"+repr(A))
            else:
                raise Exception("match error")
        return data

    @matcher(globals(), debug=False)
    def generate_server1(self, node, data, symbol_table):
        """
        SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER
        """
        def gen(node):         return self.generate_server1(node, data, symbol_table)
        def gen1(node, data1): return self.generate_server1(node, data1, symbol_table)

        if not symbol_table:
            raise Exception()

        with match(node):
            if (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                pass#self.generate_server_method(symbol_table, node, data)

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                ci = self.ClassInfo(ChapelFile(), CFile(), EPV(Name, symbol_table),
                                    ior=CFile(), skel=CFile())
                ci.chpl_stub.cstub.genh(ir.Import(Name+'_IOR'))
                self.gen_default_methods(symbol_table, Name, ci)
                gen1(Methods, ci)
                self.generate_ior(ci)

                # IOR
                write_to(Name+'_IOR.h', ci.ior.dot_h(Name+'_IOR.h'))
                write_to(Name+'_IOR.c', ci.ior.dot_c())

                # The server-side stub is used for, e.g., the
                # babelized Array-init functions

                # Stub (in C)
                cstub = ci.chpl_stub.cstub
                cstub.gen(ir.Import(Name+'_cStub'))
                cstub.gen(ir.Import('stdint'))                
                # Stub Header
                write_to(Name+'_cStub.h', cstub.dot_h(Name+'_cStub.h'))
                # Stub C-file
                write_to(Name+'_cStub.c', cstub.dot_c())

                # Skeleton (in C)
                ci.skel.gen(ir.Import(Name+'_Skel'))
                # Skel Header
                write_to(Name+'_Skel.h', ci.skel.dot_h(Name+'_Skel.h'))
                # Skel C-file
                write_to(Name+'_Skel.c', ci.skel.dot_c())

                # Impl
                write_to(Name+'_Impl.chpl', str(ci.impl))

                # Makefile
                if self.create_makefile:
                    generate_server_makefile(self.sidl_file, Name)


            elif (sidl.package, Name, Version, UserTypes, DocComment):
                self.generate_server1(UserTypes, data, symbol_table[Name])

            elif (sidl.user_type, Attrs, Cipse):
                gen(Cipse)

            elif (sidl.file, Requires, Imports, UserTypes):
                gen(UserTypes)

            elif A:
                if (isinstance(A, list)):
                    for defn in A:
                        gen(defn)
                else:
                    raise Exception("NOT HANDLED:"+repr(A))
            else:
                raise Exception("match error")
        return data


    def gen_default_methods(self, symbol_table, name, data):
        """
        Generate default Babel object methods such as _cast() Also
        generates other IOR data structures such as the _object and
        the controls and statistics struct.
        """

        def unscope((struct, scoped_id, items, doc)):
            return struct, c_gen(scoped_id), items, doc

        def builtin(t, name, args):
            data.epv.add_method(
                sidl.Method(t, sidl.Method_name(name, ''), [],
                            args, [], [], [], [], 
                            'Implicit built-in method: '+name))

        def static_builtin(t, name, args):
            data.epv.add_method(
                sidl.Method(t, sidl.Method_name(name, ''), [sidl.static],
                            args, [], [], [], [], 
                            'Implicit built-in method: '+name))

        def inarg(t, name):
            return sidl.Arg([], sidl.in_, t, name)

        # Implicit Built-in methods
        builtin(sidl.void, "_cast",
                [inarg(sidl.pt_string, 'name')])

        builtin(sidl.void, "_delete", [])

        builtin(sidl.void, "_exec", [
                inarg(sidl.pt_string, 'methodName'),
                inarg(sidl.void, 'FIXMEinArgs'),
                inarg(sidl.void, 'FIXMEoutArgs')])

        builtin(sidl.pt_string, "_getURL", [])
        builtin(sidl.void, "_raddRef", [])
        builtin(sidl.pt_bool, "_isRemote", [])
        builtin(sidl.void, '_setHooks', 
                [inarg(sidl.pt_bool, 'enable')])
        builtin(sidl.void, '_set_contracts', [
                inarg(sidl.pt_bool, 'enable'),
                inarg(sidl.pt_string, 'enfFilename'),
                inarg(sidl.pt_bool, 'resetCounters')],
                )
        builtin(sidl.void, '_dump_stats', 
                [inarg(sidl.pt_string, 'filename'),
                 inarg(sidl.pt_string, 'prefix')])
        builtin(sidl.void, '_ctor', [])
        builtin(sidl.void, '_ctor2',
                [inarg(sidl.void, 'private_data')])
        builtin(sidl.void, '_dtor', [])
        builtin(sidl.void, '_load', [])
        builtin(sidl.void, '_addRef', [])
        builtin(sidl.void, '_deleteRef', [])
        builtin(sidl.pt_bool, '_isSame',
                [inarg(babel_exception_type(), 'iobj')])
        builtin(sidl.pt_bool, '_isType',
                [inarg(sidl.pt_string, 'type')])
        builtin(babel_object_type(['sidl'], 'ClassInfo'), '_getClassInfo', [])

        static_builtin(sidl.void, '_setHooks_static', 
                [inarg(sidl.pt_bool, 'enable')])
        static_builtin(sidl.void, '_set_contracts_static', [
                inarg(sidl.pt_bool, 'enable'),
                inarg(sidl.pt_string, 'enfFilename'),
                inarg(sidl.pt_bool, 'resetCounters')],
                )
        static_builtin(sidl.void, '_dump_stats_static', 
                [inarg(sidl.pt_string, 'filename'),
                 inarg(sidl.pt_string, 'prefix')])

        # cstats
        prefix = data.epv.symbol_table.prefix
        data.cstats = \
            ir.Struct(ir.Scoped_id(prefix+[data.epv.name,'_cstats'], ''),
                      [ir.Struct_item(ir.Typedef_type("sidl_bool"), "use_hooks")],
                       'The controls and statistics structure')

        # @class@__object
        data.obj = \
            ir.Struct(ir.Scoped_id(prefix+[data.epv.name,'_object'], ''),
                      [ir.Struct_item(ir.Struct('sidl_BaseClass__object', [],''),
                                      "d_sidl_baseclass"),
                       ir.Struct_item(ir.Pointer_type(unscope(data.epv.get_type())), "d_epv"),
                       ir.Struct_item(unscope(data.cstats), "d_cstats"),
                       ir.Struct_item(ir.Pointer_type(ir.pt_void), "d_data")
                       ],
                       'The class object structure')
        data.external = \
            ir.Struct(ir.Scoped_id(prefix+[data.epv.name,'_external'], ''),
                      [ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Pointer_type(data.obj),
                                                       "createObject", [
                                                           ir.Arg([], ir.inout, ir.void_ptr, 'ddata'),
                                                           ir.Arg([], sidl.inout, ir_babel_exception_type(), 'ex')],
                                                       "")),
                                                       "createObject"),
                       # FIXME: only if contains static methods
                       ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Pointer_type(unscope(data.epv.get_sepv_type())),
                                                       "getStaticEPV", [], "")),
                                                       "getStaticEPV"),

                       ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Struct('sidl_BaseClass__epv', [],''),
                                                       "getSuperEPV", [], "")),
                                                       "getSuperEPV"),
                       ir.Struct_item(ir.pt_int, "d_ior_major_version"),
                       ir.Struct_item(ir.pt_int, "d_ior_minor_version")
                       ],
                       'The static class object structure')

    @matcher(globals(), debug=False)
    def generate_client_method(self, symbol_table, method, ci):
        """
        Generate client code for a method interface.
        \param method        s-expression of the method's SIDL declaration
        \param symbol_table  the symbol table of the SIDL file
        \param ci            a ClassInfo object
        """

        def drop_rarray_ext_args(args):
            """
            Now here it's becoming funny: Since R-arrays are wrapped inside
            SIDL-Arrays in the IOR, convention says that we should remove all
            redundant arguments that can be derived from the SIDL-array's
            metadata.
         
            @BUGS: does not yet deal with nested expressions.
            """
            names = set()
            for (arg, attrs, mode, typ, name) in args:
                if typ[0] == sidl.rarray:
                    names.update(typ[3])
         
            return filter(lambda a: a[4] not in names, args)


        def low(sidl_term):
            return lower_ir(symbol_table, sidl_term)

        def convert_arg((arg, attrs, mode, typ, name)):
            """
            Extract name and generate argument conversions
            """
            cname = name
            ctype = typ

            if is_obj_type(symbol_table, typ):
                ctype = ior_type(symbol_table, typ)

                if mode <> sidl.out:
                    cname = name+'.ior'

                if mode <> sidl.in_:
                    cname = '_IOR_'+name
                    pre_call.append(ir.Stmt(ir.Var_decl(ctype, cname)))
                    # wrap the C type in a native Chapel object
                    conv = (ir.new, ".".join(typ[1]), [cname])
                    if name == 'retval':
                        return_expr.append(conv)
                    else:
                        post_call.append(ir.Stmt(ir.Assignment(name, conv)))

            
            elif typ[0] == sidl.scoped_id:
                # Symbol
                ctype = symbol_table[typ[1]]

            elif typ == sidl.void:
                ctype = ir.pt_void

            elif typ == sidl.opaque:
                ctype = ir.Pointer_type(ir.pt_void)

            elif typ == (sidl.array, [], [] ,[]): # Generic array
                return convert_arg((arg, attrs, mode, sidl.opaque, name)) #FIXME

            elif typ[0] == sidl.array: # Scalar_type, Dimension, Orientation
                if typ[1][0] == ir.scoped_id:
                    t = 'BaseInterface'
                else:
                    t = typ[1][1]
                ctype = ir.Typedef_type('sidl_%s__array'%t)
                if mode <> sidl.out:
                    cname = name+'.ior'

                if mode <> sidl.in_:
                    cname = '_IOR_'+name
                    # wrap the C type in a native Chapel object
                    pre_call.append(ir.Stmt(ir.Var_decl(ctype, cname)))
                    if mode == sidl.inout:
                        pre_call.append(ir.Stmt(ir.Assignment(cname, name+'.ior')))

                    conv = (ir.new, 'sidl.Array', [typ[1], 'sidl_%s__array'%t, cname])
                    
                    if name == 'retval':
                        return_expr.append(conv)
                    else:
                        post_call.append(ir.Stmt(ir.Assignment(name, conv)))

            elif typ[0] == sidl.rarray: # Scalar_type, Dimension, ExtentsExpr
                # mode is always inout for an array
                original_mode = mode
                #mode = sidl.inout
                # get the type of the scalar element
                #convert_el_res = convert_arg((arg, attrs, mode, typ[1], name))
                #print convert_el_res
                ctype = ir.Typedef_type('sidl_%s__array'%typ[1][1])
                arg_name = name# convert_el_res[0]
                chpl_data_var_name = chpl_data_var_template.format(arg_name=arg_name)
                chpl_dom_var_name = chpl_dom_var_template.format(arg_name=arg_name)
                chpl_local_var_name = chpl_local_var_template.format(arg_name=arg_name)
                
                # sanity check on input arra:yensure domain is rectangular
                pre_call.append(ir.Stmt(ir.Call("performSanityCheck",
                    [chpl_dom_var_name, '"{arg_name}"'.format(arg_name=arg_name)])))
                # ensure we are working with a 'local' array
                # FIXME Hack to define a variable without explicitly specifying type
                # should we change the IR to support this?
                pre_call.append(ir.Stmt(ir.Assignment('var ' + chpl_data_var_name,
                    ir.Call("getOpaqueData", [ir.Call(arg_name, [chpl_dom_var_name + ".low"])])))
                )
                # pre_call.append(ir.Stmt(ir.Call("printAddress", [chpl_data_var_name])))
                pre_call.append(ir.Stmt(ir.Assignment('var ' + chpl_local_var_name,
                    ir.Call("ensureLocalArray", [arg_name, chpl_data_var_name])))
                )
                pre_call.append(ir.Stmt(ir.Call("checkArraysAreEqual", [arg_name, chpl_local_var_name])))

                if original_mode <> sidl.in_:
                    # emit code to copy back elements into non-local array
                    chpl_wrapper_ior_name = "_babel_wrapped_local_{arg}".format(arg=arg_name)
                    chpl_wrapper_sarray_name = "_babel_wrapped_local_{arg}_sarray".format(arg=arg_name)
                    chpl_wrapper_barray_name = "_babel_wrapped_local_{arg}_barray".format(arg=arg_name)
                    post_call.append(ir.Stmt(ir.Assignment('var ' + chpl_wrapper_sarray_name,
                        ir.Call("new Array", ["{arg}.eltType".format(arg=arg_name),
                                 ctype[1], chpl_wrapper_ior_name]))))
                    post_call.append(ir.Stmt(ir.Assignment('var ' + chpl_wrapper_barray_name,
                        ir.Call("createBorrowedArray{dim}d".format(dim=typ[2]), [chpl_wrapper_sarray_name]))))
                    post_call.append(ir.Stmt(ir.Call('syncNonLocalArray',
                        [chpl_wrapper_barray_name, arg_name])))
                    
                # Babel is strange when it comes to Rarrays. The
                # convention is to wrap Rarrays inside of a SIDL-Array
                # in the Stub and to unpack it in the
                # Skeleton. However, in some languages (C, C++) we
                # could just directly call the Impl and save all that
                # wrapper code.
                #
                # I found out the the reason behind this is that
                # r-arrays were never meant to be a performance
                # enhancement, but syntactic sugar around the SIDL
                # array implementation to offer a more native
                # interface.
                #
                # TODO: Change the IOR in Babel to use the r-array
                # calling convention.  This will complicate the
                # Python/Fortran/Java backends but simplify the C/C++
                # ones.
                sidl_wrapping = (ir.stmt, """
            var {a}rank = _babel_dom_{arg}.rank;

            var {a}lus = computeLowerUpperAndStride(_babel_local_{arg});
            var {a}lower = {a}lus(0);
            var {a}upper = {a}lus(1);
            var {a}stride = {a}lus(2);
            
            var _babel_wrapped_local_{arg}: {ctype} = {ctype}_borrow(
                {stype}_ptr(_babel_local_{arg}(_babel_local_{arg}.domain.low)),
                {a}rank,
                {a}lower[1],
                {a}upper[1],
                {a}stride[1])""".format(a='_babel_%s_'%arg_name,
                                         arg=arg_name,
                                         ctype=ctype[1],
                                         stype=typ[1][1]))
                
                pre_call.append(sidl_wrapping)
                post_call.append((ir.stmt, '//sidl__array_deleteRef((struct sidl__array*)a_tmp)'))

                # reference the lowest element of the array using the domain
                #call_expr_str = chpl_local_var_name + '(' + chpl_local_var_name + '.domain.low' + ')'
                #return (call_expr_str, convert_el_res[1])
                return '_babel_wrapped_local_'+arg_name, (arg, attrs, mode, ctype, name)
                

            return cname, (arg, attrs, mode, ctype, name)

        def obj_by_value((arg, attrs, mode, typ, name)):
            if is_obj_type(symbol_table, typ):
                return (arg, attrs, sidl.in_, typ, name)
            else:
                return (arg, attrs, mode, typ, name)

        # Chapel stub
        (Method, Type, (MName,  Name, Extension), Attrs, Args,
         Except, From, Requires, Ensures, DocComment) = method

        ior_args = drop_rarray_ext_args(Args)

        ci.epv.add_method((Method, Type, (MName,  Name, Extension), Attrs, ior_args,
                           Except, From, Requires, Ensures, DocComment))

        abstract = list(member(sidl.abstract, Attrs))
        static = list(member(sidl.static, Attrs))
        final = list(member(sidl.static, Attrs))

        if abstract:
            # nothing to be done for an abstract function
            return

        pre_call = [ir.Stmt(ir.Var_decl(ir_babel_exception_type(), 'ex'))]
        post_call = []
        call_args, cdecl_args = unzip(map(convert_arg, ior_args))
        return_expr = []
        return_stmt = []

        # return value type conversion -- treat it as an out argument
        _, (_,_,_,ctype,_) = convert_arg((ir.arg, [], ir.out, Type, 'retval'))

        cdecl_args = babel_stub_args(Attrs, cdecl_args, symbol_table, ci.epv.name)
        cdecl = ir.Fn_decl(Attrs, ctype, Name+Extension, cdecl_args, DocComment)

        if static:
            extern_self = []
            call_self = []
        else:
            extern_self = [ir.Arg([], ir.inout, ir_babel_object_type(
                symbol_table.prefix, ci.epv.name), 'self')]
            call_self = ["self"]

        call_args = call_self + call_args + ['ex']

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
                ir.Struct_item(ir.Pointer_type(cdecl), 'f_'+Name+Extension))
            
        else:
            # dynamic virtual method call
            epv_type = ci.epv.get_type()
            obj_type = ci.obj
            callee = ir.Deref(ir.Get_struct_item(
                epv_type,
                ir.Deref(ir.Get_struct_item(obj_type,
                                            ir.Deref('self'),
                                            ir.Struct_item(epv_type, 'd_epv'))),
                ir.Struct_item(ir.Pointer_type(cdecl), 'f_'+Name+Extension)))


        if Type == sidl.void:
            Type = ir.pt_void
            call = [ir.Stmt(ir.Call(callee, call_args))]
        else:
            if return_expr or post_call:
                rvar = '_IOR_retval'
                if not return_expr:
                    pre_call.append(ir.Stmt(ir.Var_decl(ctype, rvar)))
                    rx = rvar
                else:
                    rx = return_expr[0]
                    
                call = [ir.Stmt(ir.Assignment(rvar, ir.Call(callee, call_args)))]
                return_stmt = [ir.Stmt(ir.Return(rx))]
            else:
                call = [ir.Stmt(ir.Return(ir.Call(callee, call_args)))]

        defn = (ir.fn_defn, [], Type, Name, Args,
                pre_call+call+post_call+return_stmt,
                DocComment)

        if static:
            # # FIXME final functions still _may_ have a cstub
            # # FIXME can we reuse cdecl for this?
            # # see the __attribute__ hack below
            # impl_decl = ir.Fn_decl([], ctype,
            #                        callee, cdecl_args, DocComment)
            # extern_decl = ir.Fn_decl([], ctype,
            #                          callee, map(obj_by_value, cdecl_args), DocComment)
            # ci.chpl_static_stub.new_def('_extern '+chpl_gen(extern_decl)+';')
            # ci.chpl_stub.cstub.new_header_def('extern '+str(c_gen(impl_decl))+';')
            chpl_gen(defn, ci.chpl_static_stub)
        else:
            chpl_gen(defn, ci.chpl_stub)

    def generate_ior(self, ci):
        """
        Generate the IOR header file in C.
        """
        ci.ior.genh(ir.Import('_'.join(ci.epv.symbol_table.prefix)))
        ci.ior.genh(ir.Import('sidl'))
        ci.ior.genh(ir.Import('sidl_BaseInterface_IOR'))
        ci.ior.genh(ir.Import('stdint'))
        ci.ior.genh(ir.Import('chpl_sidl_array'))
        ci.ior.gen(ir.Type_decl(ci.cstats))
        ci.ior.gen(ir.Type_decl(ci.obj))
        ci.ior.gen(ir.Type_decl(ci.external))
        ci.ior.gen(ir.Type_decl(ci.epv.get_ir()))
        ci.ior.gen(ir.Type_decl(ci.epv.get_sepv_ir()))


char_lut = '''
/* This burial ground of bytes is used for char [in]out arguments. */
static const unsigned char chpl_char_lut[512] = {
  '''+' '.join(['%d, 0,'%i for i in range(0, 256)])+'''
};
'''

def externals(prefix):
    return '''
// Hold pointer to IOR functions.
static const struct {a}__external *_externals = NULL;

// Lookup the symbol to get the IOR functions.
static const struct {a}__external* _loadIOR(void)

// Return pointer to internal IOR functions.
{{
#ifdef SIDL_STATIC_LIBRARY
  _externals = {a}__externals();
#else
  _externals = (struct {a}__external*)sidl_dynamicLoadIOR(
    "ArrayTest.ArrayOps","{a}__externals") ;
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

'''.format(a='_'.join(prefix), b='.'.join(prefix))

def generate_method_stub(scope, (_call, VCallExpr, CallArgs), prefix):
    """
    Generate the stub for a specific method in C (cStub).

    \return   if the methods needs an extra inout argument for the
              return value, this function returns its type, otherwise
              it returns \c None.
    """
    def convert_arg((arg, attrs, mode, typ, name)):
        """
        Extract name and generate argument conversions
        """
        call_name = name
        deref = ref = ''
        accessor = '.'
        if mode <> sidl.in_ and name <> '_retval':
            deref = '*'
            ref = '&'
            accessor = '->'

        # Case-by-case for each data type

        # BOOL
        if typ == sidl.pt_bool:
            pre_call.append(ir.Comment(
                'sidl_bool is an int, but chapel bool is a char/_Bool'))
            pre_call.append((ir.stmt, 'int _babel_'+name))

            if mode <> sidl.out:
                pre_call.append((ir.stmt, '_babel_{n} = (int){p}{n}'
                                 .format(n=name, p=deref)))

            if mode <> sidl.in_:
                post_call.append((ir.stmt, '{p}{n} = ({typ})_babel_{n}'
                        .format(p=deref, n=name, typ=c_gen(sidl.pt_bool))))

            call_name = ref+'_babel_'+name
            # Bypass the bool -> int conversion for the stub decl
            typ = (ir.typedef_type, '_Bool')

        # CHAR
        elif typ == sidl.pt_char:
            pre_call.append(ir.Comment(
                'in chapel, a char is a string of length 1'))
            typ = ir.const_str

            pre_call.append((ir.stmt, 'char _babel_%s'%name))
            if mode <> sidl.out:
                pre_call.append((ir.stmt, '_babel_{n} = (int){p}{n}[0]'
                                 .format(n=name, p=deref)))

            if mode <> sidl.in_:
                # we can't allocate a new string, this would leak memory
                post_call.append((ir.stmt,
                    '{p}{n} = (const char*)&chpl_char_lut[2*(unsigned char)_babel_{n}]'
                                  .format(p=deref, n=name)))
            call_name = ref+'_babel_'+name
            scope.cstub.optional.add(char_lut)

        # STRING
        elif typ == sidl.pt_string:
            typ = ir.const_str
            if mode <> sidl.in_:
                # Convert null pointer into empty string
                post_call.append((ir.stmt, 'if ({p}{n} == NULL) {p}{n} = ""'
                                  .format(n=name, p=deref))) 

        # INT
        elif typ == sidl.pt_int:
            typ = ir.Typedef_type('int32_t')

        # LONG
        elif typ == sidl.pt_long:
            typ = ir.Typedef_type('int64_t')
        
        # COMPLEX - 32/64 Bit components
        elif (typ == sidl.pt_fcomplex or typ == sidl.pt_dcomplex):
            
            sidl_type_str = 'struct '+('sidl_fcomplex' if (typ == sidl.pt_fcomplex)
                                  else 'sidl_dcomplex')
            complex_type_name = '_complex64' if (typ == sidl.pt_fcomplex) else '_complex128'
            
            pre_call.append(ir.Comment(
                'in chapel, a ' + sidl_type_str + ' is a ' + complex_type_name))
            typ = ir.Typedef_type(complex_type_name)
            call_name = ref + '_babel_' + name
            pre_call.append((ir.stmt, '{t} _babel_{n}'.format(t=sidl_type_str, n=name)))
            fmt = {'n':name, 'a':accessor}
            if mode <> sidl.out:
                pre_call.append((ir.stmt, '_babel_{n}.real = {n}{a}re'.format(**fmt)))
                pre_call.append((ir.stmt, '_babel_{n}.imaginary = {n}{a}im'.format(**fmt)))
            if mode <> sidl.in_:
                post_call.append((ir.stmt, '{n}{a}re = _babel_{n}.real'.format(**fmt)))
                post_call.append((ir.stmt, '{n}{a}im = _babel_{n}.imaginary'.format(**fmt)))
            
        elif typ[0] == sidl.enum:
            call_name = ir.Sign_extend(64, name)
            
        # SELF
        # cf. babel_stub_args
        #elif name == 'self':
        #    typ = typ[1]
        #    #call_name = '*self'
        #elif typ[0] == ir.pointer_type and typ[1][0] == ir.struct: # ex
        #    typ = typ[1]

        # ARRAYS
        elif typ[0] == sidl.array: # Scalar_type, Dimension, Orientation
            import pdb; pdb.set_trace()

        return call_name, (arg, attrs, mode, typ, name)

    def obj_by_value((arg, attrs, mode, typ, name)):
        if typ[0] == ir.struct and name == 'self':
            return (arg, attrs, sidl.in_, typ, name)
        else:
            return (arg, attrs, mode, typ, name)


    if VCallExpr[0] == ir.deref:
        _, (_, _ , _, (_, (_, impl_decl), _)) = VCallExpr
    else:
        _, _ , _, (_, (_, impl_decl), _) = VCallExpr

    (_, Attrs, Type, Name, Args, DocComment) = impl_decl
    sname = '_'.join(prefix+[Name, 'stub'])

    retval_arg = []
    pre_call = []
    post_call = []
    call_args, cstub_decl_args = unzip(map(convert_arg, Args))

    # return value type conversion -- treat it as an out argument
    retval_expr, (_,_,_,ctype,_) = convert_arg((ir.arg, [], ir.out, Type, '_retval'))
    cstub_decl = ir.Fn_decl([], ctype, sname, cstub_decl_args, DocComment)

    static = list(member(sidl.static, Attrs))
    if static:
        scope.cstub.optional.add(externals(prefix))

    if Type == ir.pt_void:
        body = [ir.Stmt(ir.Call(VCallExpr, call_args))]
    else:
        if Type == sidl.pt_char:
            pre_call.append(ir.Stmt(ir.Var_decl(ir.Typedef_type('const char*'), '_retval')))
        elif Type == sidl.pt_fcomplex:
            pre_call.append(ir.Stmt(ir.Var_decl(ir.Typedef_type('_complex64'), '_retval')))
        elif Type == sidl.pt_dcomplex:
            pre_call.append(ir.Stmt(ir.Var_decl(ir.Typedef_type('_complex128'), '_retval')))
        else:
            pre_call.append(ir.Stmt(ir.Var_decl(Type, '_retval')))
        body = [ir.Stmt(ir.Assignment(retval_expr,
                                      ir.Call(VCallExpr, call_args)))]
        post_call.append(ir.Stmt(ir.Return('_retval')))

    # Generate the C code into the scope's associated cStub
    c_gen([cstub_decl,
           ir.Fn_defn([], ctype, sname, cstub_decl_args,
                      pre_call+body+post_call, DocComment)], scope.cstub)
    
    # Chapel _extern declaration
    chplstub_decl = ir.Fn_decl([], ctype, sname, map(obj_by_value, cstub_decl_args), DocComment)
    scope.get_toplevel().new_header_def('_extern '+chpl_gen(chplstub_decl))

    return drop(retval_arg)

def is_obj_type(symbol_table, typ):
    return typ[0] == sidl.scoped_id and (
        symbol_table[typ[1]][0] == sidl.class_ or
        symbol_table[typ[1]][0] == sidl.interface)


def ior_type(symbol_table, t):
    """
    if \c t is a scoped_id return the IOR type of t.
    else return \c t.
    """
    if (t[0] == sidl.scoped_id and
        symbol_table[t[1]][0] == sidl.class_):
        return ir.Pointer_type(ir_babel_object_type(*symbol_table.get_full_name(t[1])))

    else: return t


@matcher(globals(), debug=False)
def lower_ir(symbol_table, sidl_term):
    """
    lower SIDL into IR
    """
    def low(sidl_term):
        return lower_ir(symbol_table, sidl_term)

    def low_t(sidl_term):
        return lower_type_ir(symbol_table, sidl_term)

    with match(sidl_term):
        if (sidl.struct, Name, Items):
            return ir.Pointer_expr(ir.Struct(low_t(Name), Items, ''))

        elif (sidl.arg, Attrs, Mode, Typ, Name):
            return ir.Arg(Attrs, Mode, low_t(Typ), Name)

        elif (sidl.void):              return ir.pt_void
        elif (sidl.primitive_type, _): return low_t(sidl_term)
        elif (sidl.scoped_id, _, _):   return low_t(sidl_term)
        elif (sidl.array, _, _, _):    return low_t(sidl_term)
        elif (sidl.rarray, _, _, _):   return low_t(sidl_term)

        elif (Terms):
            if (isinstance(Terms, list)):
                return map(low, Terms)
        else:
            raise Exception("Not implemented")

@matcher(globals(), debug=False)
def lower_type_ir(symbol_table, sidl_type):
    """
    FIXME!! can we merge this with convert_arg??
    lower SIDL types into IR
    """
    with match(sidl_type):
        if (sidl.scoped_id, Names, Ext):
            return lower_type_ir(symbol_table, symbol_table[Names])
        
        elif (sidl.void):                        return ir.pt_void
        elif (sidl.primitive_type, sidl.opaque): return ir.Pointer_type(ir.pt_void)
        elif (sidl.primitive_type, sidl.string): return ir.const_str
        elif (sidl.primitive_type, sidl.bool):   return ir.pt_int
        elif (sidl.primitive_type, sidl.long):   return ir.Typedef_type('int64_t')
        elif (sidl.primitive_type, Type):        return ir.Primitive_type(Type)
        elif (sidl.enum, _, _, _):               return ir.Typedef_type('int64_t')
        elif (sidl.enumerator, _):               return sidl_type # identical
        elif (sidl.enumerator, _, _):            return sidl_type
        elif (sidl.rarray, Scalar_type, Dimension, Extents):
            # Direct-call version (r-array IOR)
            # return ir.Pointer_type(lower_type_ir(symbol_table, Scalar_type)) # FIXME
            # SIDL IOR version
            return ir.Typedef_type('sidl_%s__array'%Scalar_type[1])

        elif (sidl.array, [], [], []):
            return ir.Pointer_type(ir.pt_void)

        elif (sidl.array, Scalar_type, Dimension, Orientation):
            #return ir.Typedef_type('sidl__array')
            if Scalar_type[0] == ir.scoped_id:
                t = 'BaseInterface'
            else:
                t = Scalar_type[1]
            return ir.Typedef_type('sidl_%s__array'%t)


        elif (sidl.class_, Name, _, _, _, _):
            return ir_babel_object_type([], Name)
        
        elif (sidl.interface, Name, _, _, _):
            return ir_babel_object_type([], Name)
        
        else:
            raise Exception("Not implemented")

def get_type_name((fn_decl, Attrs, Type, Name, Args, DocComment)):
    return ir.Pointer_type((fn_decl, Attrs, Type, Name, Args, DocComment)), Name

class EPV:
    """
    Babel entry point vector for virtual method calls.
    Also contains the SEPV which is used for all static functions.
    """
    def __init__(self, name, symbol_table):
        self.methods = []
        self.static_methods = []
        self.name = name
        self.symbol_table = symbol_table
        self.finalized = False

    def add_method(self, method):
        """
        add another (SIDL) method to the vector
        """
        def to_fn_decl((_sidl_method, Type,
                        (Method_name, Name, Extension),
                        Attrs, Args, Except, From, Requires, Ensures, DocComment)):
            typ = lower_ir(self.symbol_table, Type)
            if typ[0] == ir.struct:
                typ = ir.Pointer_type(typ)
            name = 'f_'+Name+Extension
            args = babel_epv_args(Attrs, Args, self.symbol_table, self.name)
            return ir.Fn_decl(Attrs, typ, name, args, DocComment)

        if self.finalized:
            import pdb; pdb.set_trace()

        if list(member(sidl.abstract, method[3])):
            pass
        elif list(member(sidl.static, method[3])):
            self.static_methods.append(to_fn_decl(method))
        else:
            self.methods.append(to_fn_decl(method))
        return self

    def get_ir(self):
        """
        return an s-expression of the EPV declaration
        """

        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix+[self.name,'_epv'], '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname)
             for itype, iname in map(get_type_name, self.methods)],
                         'Entry Point Vector (EPV)')

    def get_sepv_ir(self):
        """
        return an s-expression of the SEPV declaration
        """

        self.finalized = True
        name = ir.Scoped_id(self.symbol_table.prefix+[self.name,'_sepv'], '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname)
             for itype, iname in map(get_type_name, self.static_methods)],
                         'Static Entry Point Vector (SEPV)')


    def get_type(self):
        """
        return an s-expression of the EPV's type
        """
        name = ir.Scoped_id(self.symbol_table.prefix+[self.name,'_epv'], '')
        return ir.Struct(name, [], 'Entry Point Vector (EPV)')

    def get_sepv_type(self):
        """
        return an s-expression of the SEPV's type
        """
        name = ir.Scoped_id(self.symbol_table.prefix+[self.name,'_sepv'], '')
        return ir.Struct(name, [], 'Static Entry Point Vector (SEPV)')



def babel_epv_args(attrs, args, symbol_table, class_name):
    """
    \return a SIDL -> Ir lowered version of [self]+args+[*ex]
    """
    if list(member(sidl.static, attrs)):
        arg_self = []
    else:
        arg_self = \
            [ir.Arg([], ir.in_, ir.Pointer_type(
                    ir_babel_object_type(symbol_table.prefix, class_name)),
                    'self')]
    arg_ex = \
        [ir.Arg([], sidl.inout, ir_babel_exception_type(), 'ex')]
    return arg_self+lower_ir(symbol_table, args)+arg_ex

def babel_stub_args(attrs, args, symbol_table, class_name):
    """
    \return a SIDL -> [*self]+args+[*ex]
    """
    if list(member(sidl.static, attrs)):
        arg_self = []
    else:
        arg_self = [
            ir.Arg([], sidl.inout, 
                ir_babel_object_type(symbol_table.prefix, class_name), 'self')]
    arg_ex = \
        [ir.Arg([], sidl.inout, ir_babel_exception_type(), 'ex')]
    return arg_self+args+arg_ex


def arg_ex():
    """
    default SIDL exception argument
    """
    return


class ChapelFile(SourceFile):
    """
    Particularities:
    
    * Chapel files also have a cstub which is used to output code that
      can not otherwise be expressed in Chapel.
    """
    
    def __init__(self, parent=None, relative_indent=0):
        super(ChapelFile, self).__init__(
            parent, relative_indent, separator='\n')
        if parent:
            self.cstub = parent.cstub
        else:
            self.cstub = CFile()
            # This is for definitions that are generated in multiple
            # locations but should be written out only once.
            self.cstub.optional = set()

    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        h_indent = ''
        d_indent = ''
        if len(self._header) > 0: h_indent=self._sep
        if len(self._defs) > 0:   d_indent=self._sep

        return ''.join([
            h_indent,
            sep_by(';'+self._sep, self._header),
            d_indent,
            self._sep.join(self._defs)])

    def get_decls(self):
        h_indent = ''
        if len(self._header) > 0: 
            h_indent=self._sep
        return ''.join([h_indent, sep_by(';'+self._sep, self._header)])

    def get_defs(self):
        d_indent = ''
        if len(self._defs) > 0:   
            d_indent=self._sep
        return ''.join([d_indent, sep_by(self._sep, self._defs)])


    def gen(self, ir):
        """
        Invoke the Chapel code generator on \c ir and append the result to
        this ChapelFile object.
        """
        ChapelCodeGenerator().generate(ir, self)


class ChapelScope(ChapelFile):
    def __init__(self, parent=None, relative_indent=4):
        super(ChapelScope, self).__init__(parent, relative_indent)

    def __str__(self):
        return self._sep.join(self._header+self._defs)

class ChapelLine(ChapelFile):
    def __init__(self, parent=None, relative_indent=4):
        super(ChapelLine, self).__init__(parent, relative_indent)

    def __str__(self):
        return self._sep.join(self._header+self._defs)

def chpl_gen(ir, scope=None):
    if scope == None:
        scope = ChapelScope()
    return str(ChapelCodeGenerator().generate(ir, scope))

def c_gen(ir, scope=None):
    if scope == None:
        scope = CFile()
    return CCodeGenerator().generate(ir, scope)

class ChapelCodeGenerator(ClikeCodeGenerator):
    type_map = {
        'void':      "void",
        'bool':      "bool",
        'char':      "string",
        'dcomplex':  "complex(128)",
        'double':    "real(64)",
        'fcomplex':  "complex(64)",
        'float':     "real(32)",
        'int':       "int(32)",
        'long':      "int(64)",
        'opaque':    "int(64)",
        'string':    "string"
        }

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=ChapelFile()):
        """
        This code generator is a bit unusual in that it accepts a
        hybrid of \c sidl and \c ir nodes.
        """
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            return scope.new_def(s)

        def new_header_def(s):
            return scope.new_header_def(s)

        @accepts(str, list, str)
        def new_scope(prefix, body, suffix='\n'):
            '''used for things like if, while, ...'''
            comp_stmt = ChapelFile(scope, relative_indent=4)
            s = str(self.generate(body, comp_stmt))
            return new_def(scope._sep.join([prefix+s,suffix]))

        def gen_comma_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=1, separator=','))

        def gen_ws_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator=' '))

        def gen_dot_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator='.'))

        def tmap(f, l):
            return tuple(map(f, l))

        def gen_comment(doc_comment):
            if doc_comment == '':
                return ''
            sep = '\n'+' '*scope.indent_level
            return (sep+' * ').join(['/**']+
                                   re.split('\n\s*', doc_comment)
                                   )+sep+' */'+sep
        cbool = '_Bool'
        int32 = 'int32_t'
        int64 = 'int64_t'
        fcomplex = '_complex64'
        dcomplex = '_complex128'
        
        val = self.generate_non_tuple(node, scope)
        if val <> None:
            return val

        with match(node):
            if (ir.fn_defn, Attrs, (ir.primitive_type, 'void'), Name, Args, Body, DocComment):
                new_scope('%sproc %s(%s) {'%
                          (gen_comment(DocComment), 
                           gen(Name), gen_comma_sep(Args)),
                          Body,
                          '}')
                new_def('')

            elif (ir.fn_defn, Attrs, Type, Name, Args, Body, DocComment):
                new_scope('%sproc %s(%s): %s {'%
                          (gen_comment(DocComment), 
                           gen(Name), gen_comma_sep(Args),
                           gen(Type)),
                          Body,
                          '}')
                new_def('')

            elif (ir.fn_decl, Attrs, (ir.primitive_type, 'void'), Name, Args, DocComment):
                new_def('proc %s(%s)'% (gen(Name), gen_comma_sep(Args)))

            elif (ir.fn_decl, Attrs, Type, Name, Args, DocComment):
                new_def('proc %s(%s): %s'%
                        (gen(Name), gen_comma_sep(Args), gen(Type)))

            elif (ir.call, (ir.deref, (ir.get_struct_item, S, _, (ir.struct_item, _, Name))), Args):
                # We can't do a function pointer call in Chapel
                # Emit a C stub for that
                _, (_, ids, _), _, _ = S
                prefix = ids[:-1]
                retval_arg = generate_method_stub(scope, node, prefix)
                stubname = '_'.join(prefix+[re.sub('^f_', '', Name),'stub'])
                if retval_arg:
                    scope.pre_def(gen(ir.Var_decl(retval_arg, '_retval')))
                    scope.pre_def(gen(ir.Assignment('_retval', ir.Call(stubname, Args+retval_arg))))
                    return '_retval'
                else:
                    return gen(ir.Call(stubname, Args))

            elif (ir.call, (ir.get_struct_item, S, _, (ir.struct_item, _, Name)), Args):
                # We can't do a function pointer call in Chapel
                # Emit a C stub for that
                _, (_, ids, _), _, _ = S
                prefix = ids[:-1]
                retval_arg = generate_method_stub(scope, node, prefix)
                stubname = '_'.join(prefix+[re.sub('^f_', '', Name),'stub'])
                if retval_arg:
                    scope.pre_def(gen(ir.Var_decl(retval_arg, '_retval')))
                    scope.pre_def(gen(ir.Assignment('_retval', ir.Call(stubname, Args+retval_arg))))
                    return '_retval'
                else:
                    return gen(ir.Call(stubname, Args))

            elif (ir.new, Type, Args):
                return 'new %s(%s)'%(gen(Type), gen_comma_sep(Args))

            elif (ir.const, Type):
                return '/*FIXME: CONST*/'+gen(Type)

            # Special handling of rarray types
            elif (sidl.arg, Attrs, Mode, (sidl.rarray, Scalar_type, Dimension, Extents), Name):
                (arg_mode, arg_name) = (gen(Mode), gen(Name))
                # rarray type will include a new domain variable definition
                chpl_dom_var_name = chpl_dom_var_template.format(arg_name=Name)
                arg_type = '[?' + chpl_dom_var_name + '] ' + gen(Scalar_type)
                return '%s %s: %s'%(arg_mode, arg_name, arg_type)
                
            elif (sidl.arg, Attrs, Mode, Type, Name):
                return '%s %s: %s'%(gen(Mode), gen(Name), gen(Type))

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, Package, DocComment):
                return gen_comment(DocComment)+'class '+Name

            elif (sidl.array, [], [], []):
                return gen(ir.pt_void)+'/*FIXME*/'

            elif (sidl.array, Scalar_type, Dimension, Orientation):
                if Scalar_type[0] == ir.scoped_id:
                    ctype = 'BaseInterface'
                else:
                    ctype = Scalar_type[1]
                return 'sidl.Array(%s, sidl_%s__array)'%(gen(Scalar_type), ctype)
                scope.cstub.optional.add('#include <sidl_%s_IOR.h>'%ctype)

            elif (ir.pointer_type, (ir.const, (ir.primitive_type, ir.char))):
                return "string"

            elif (ir.pointer_type, Type):
                # ignore wrongfully introduced pointers
                # -> actually I should fix generate_method_stub instead
                return gen(Type)

            elif (ir.typedef_type, cbool):
                return "bool"

            elif (ir.typedef_type, int32):
                return "int(32)"

            elif (ir.typedef_type, int64):
                return "int(64)"
            
            elif (ir.typedef_type, fcomplex):
                return "complex(64)"
            
            elif (ir.typedef_type, dcomplex):
                return "complex(128)"
            
            elif (ir.struct, (ir.scoped_id, Names, Ext), Items, DocComment):
                return '_'.join(Names)

            elif (ir.var_decl, Type, Name):
                return 'var %s:%s'%(gen(Name), gen(Type))

            elif (ir.var_decl_init, (ir.typedef_type, "inferred_type"), Name, Initializer): 
                return 'var %s = %s'%(gen(Name), gen(Initializer))

            elif (ir.var_decl_init, Type, Name, Initializer): 
                return 'var %s:%s = %s'%(gen(Name), gen(Type), gen(Initializer))

            elif (ir.enum, Name, Items, DocComment): return gen(Name)

            elif (sidl.custom_attribute, Id):       return gen(Id)
            elif (sidl.method_name, Id, Extension): return gen(Id)
            elif (sidl.scoped_id, A, B):
                return '%s%s' % (gen_dot_sep(A), gen(B))

            elif (Expr):
                return super(ChapelCodeGenerator, self).generate(Expr, scope)

            else:
                raise Exception('match error')
        return scope

def generate_client_makefile(sidl_file, classes):
    """
    FIXME: make this a file copy from $prefix/share
           make this work for more than one class
    """
    files = 'IORHDRS = '+' '.join([c+'_IOR.h' for c in classes])+'\n'
    files+= 'STUBHDRS = '+' '.join(['{c}.h {c}_Stub.h {c}_cStub.h'.format(c=c)
                                    for c in classes])+'\n'
    files+= 'STUBSRCS = '+' '.join([c+'_cStub.c' for c in classes])+'\n'
# this is handled by the use statement in the implementation instead:
# {file}_Stub.chpl
    write_to('babel.make', files)
    generate_client_server_makefile(sidl_file)


def generate_server_makefile(sidl_file, classnames):
    """
    FIXME: make this a file copy from $prefix/share
           make this work for more than one class
    """
    write_to('babel.make', """
IMPLHDRS =
IMPLSRCS = {file}_Impl.chpl
IORHDRS = {file}_IOR.h #FIXME Array_IOR.h
IORSRCS = {file}_IOR.c
SKELSRCS = {file}_Skel.c
STUBHDRS = #FIXME {file}.h
STUBSRCS = {file}_cStub.c
""".format(file=classnames))
    generate_client_server_makefile(sidl_file)

def generate_client_server_makefile(sidl_file):
    write_to('GNUmakefile', r"""
# Generic Chapel Babel wrapper GNU Makefile
# $Id$
#
# Copyright (c) 2008, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the Components Team <components@llnl.gov>
# UCRL-CODE-2002-054
# All rights reserved.
#
# This file is part of Babel. For more information, see
# http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
# for Our Notice and the LICENSE file for the GNU Lesser General Public
# License.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
#
# You should have recieved a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# This Makefile uses GNU make extensions, so it may not work with
# other implementations of make.

include babel.make
# please name the server library here
LIBNAME=impl
# please name the SIDL file here
SIDLFILE="""+sidl_file+r"""
# extra include/compile flags
EXTRAFLAGS=-ggdb -O0
# extra libraries that the implementation needs to link against
EXTRALIBS=
# library version number
VERSION=0.1.1
# PREFIX specifies the top of the installation directory
PREFIX=/usr/local
# the default installation installs the .la and .scl (if any) into the
# LIBDIR
LIBDIR=$(PREFIX)/lib
# the default installation installs the stub header and IOR header files
# in INCLDIR
INCLDIR=$(PREFIX)/include
CHAPEL="""+config.CHAPEL+r"""
CHAPEL_ROOT="""+config.CHAPEL_ROOT+r"""
CHAPEL_MAKE_MEM=default
# CHAPEL_MAKE_COMM=none
CHAPEL_MAKE_COMM="""+config.CHAPEL_COMM+r"""
CHAPEL_MAKE_COMPILER=gnu
CHAPEL_MAKE_TASKS=fifo
CHAPEL_MAKE_THREADS=pthreads

ifeq ($(CHAPEL_MAKE_COMM),gasnet)
CHAPEL_MAKE_SUBSTRATE_DIR=$(CHAPEL_ROOT)/lib/$(CHPL_HOST_PLATFORM)/$(CHAPEL_MAKE_COMPILER)/comm-gasnet-nodbg/substrate-udp
else
CHAPEL_MAKE_SUBSTRATE_DIR=$(CHAPEL_ROOT)/lib/$(CHPL_HOST_PLATFORM)/$(CHAPEL_MAKE_COMPILER)/comm-none/substrate-none
endif
####    include $(CHAPEL_ROOT)/runtime/etc/Makefile.include
CHPL=chpl
# CHPL=chpl --print-commands --print-passes

CHPL_FLAGS=-std=c99 -DCHPL_TASKS_H=\"tasks-fifo.h\" -DCHPL_THREADS_H=\"threads-pthreads.h\" -I$(CHAPEL_ROOT)/runtime/include/tasks/fifo -I$(CHAPEL_ROOT)/runtime/include/threads/pthreads -I$(CHAPEL_ROOT)/runtime/include/comm/none -I$(CHAPEL_ROOT)/runtime/include/comp-gnu -I$(CHAPEL_ROOT)/runtime/include/$(CHPL_HOST_PLATFORM) -I$(CHAPEL_ROOT)/runtime/include -I. -Wno-all

CHPL_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm  -lpthread

CHPL_GASNET_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm -lpthread -L$(CHAPEL_ROOT)/third-party/gasnet/install/$(CHPL_HOST_PLATFORM)-$(CHAPEL_MAKE_COMPILER)/seg-everything/nodbg/lib -lgasnet-udp-par -lamudp -lpthread -lgcc -lm

CHPL_LAUNCHER_LDFLAGS=$(CHAPEL_MAKE_SUBSTRATE_DIR)/launch-amudprun/main_launcher.o
LAUNCHER_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads -L$(CHAPEL_MAKE_SUBSTRATE_DIR)/launch-amudprun -lchpllaunch -lchpl -lm

SIDL_RUNTIME="""+config.PREFIX+r"""/include
CHPL_HEADERS=-I$(SIDL_RUNTIME)/chpl -M$(SIDL_RUNTIME)/chpl \
  chpl_sidl_array.h

# most of the rest of the file should not require editing

ifeq ($(IMPLSRCS),)
  SCLFILE=
  BABELFLAG=--client=Chapel
  MODFLAG=
else
  SCLFILE=lib$(LIBNAME).scl
  BABELFLAG=--server=Chapel
  MODFLAG=-module
endif

ifeq ($(CHAPEL_MAKE_COMM),gasnet)

all: lib$(LIBNAME).la $(SCLFILE) $(OUTFILE) $(OUTFILE)_real

# actual program
$(OUTFILE)_real: lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
	babel-libtool --mode=link $(CXX) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) \
          $(CHPL_GASNET_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

# launcher
$(OUTFILE): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo
	echo "#include \"chplcgfns.h\"" > $(IMPL).chpl.dir/config.c
	echo "#include \"config.h\""   >> $(IMPL).chpl.dir/config.c
	echo "#include \"_config.c\""  >> $(IMPL).chpl.dir/config.c
	babel-libtool --mode=compile --tag=CC $(CC) \
          -std=c99 -I$(CHAPEL_ROOT)/runtime/include/$(CHPL_HOST_PLATFORM) \
	  -I$(CHAPEL_ROOT)/runtime/include -I. \
	  $(IMPL).chpl.dir/config.c -c -o $@.lo
	babel-libtool --mode=link $(CC) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $@.lo $(SERVER) \
          $(CHPL_LAUNCHER_LDFLAGS) $(LAUNCHER_LDFLAGS) $(EXTRA_LDFLAGS) -o $@

else

all: lib$(LIBNAME).la $(SCLFILE) $(OUTFILE)

$(OUTFILE): lib$(LIBNAME).la $(SERVER) $(IMPLOBJS) $(IMPL).lo 
	babel-libtool --mode=link $(CC) -static lib$(LIBNAME).la \
	  $(IMPLOBJS) $(IMPL).lo $(SERVER) $(CHPL_LDFLAGS) $(EXTRA_LDFLAGS) -o $@
endif

CC=`babel-config --query-var=CC`
INCLUDES=`babel-config --includes` -I. -I$(CHAPEL_ROOT)/runtime/include -I$(SIDL_RUNTIME)/chpl
CFLAGS=`babel-config --flags-c` -std=c99
LIBS=`babel-config --libs-c-client`

STUBOBJS=$(patsubst .chpl, .lo, $(STUBSRCS:.c=.lo))
IOROBJS=$(IORSRCS:.c=.lo)
SKELOBJS=$(SKELSRCS:.c=.lo)
IMPLOBJS=$(IMPLSRCS:.chpl=.lo)

PUREBABELGEN=$(IORHDRS) $(IORSRCS) $(STUBSRCS) $(STUBHDRS) $(SKELSRCS)
BABELGEN=$(IMPLHDRS) $(IMPLSRCS)

$(IMPLOBJS) : $(STUBHDRS) $(IORHDRS) $(IMPLHDRS)

lib$(LIBNAME).la : $(STUBOBJS) $(IOROBJS) $(IMPLOBJS) $(SKELOBJS)
	babel-libtool --mode=link --tag=CC $(CC) -o lib$(LIBNAME).la \
	  -rpath $(LIBDIR) -release $(VERSION) \
	  -no-undefined $(MODFLAG) \
	  $(CFLAGS) $(EXTRAFLAGS) $^ $(LIBS) \
	  $(EXTRALIBS)

$(PUREBABELGEN) $(BABELGEN) : babel-stamp
	@if test -f $@; then \
	    touch $@; \
	else \
	    rm -f babel-stamp ; \
	    $(MAKE) babel-stamp; \
	fi

babel-stamp: $(SIDLFILE)
	@rm -f babel-temp
	@touch babel-temp
	braid $(BABELFLAG) $(SIDLFILE)
	@mv -f babel-temp $@

lib$(LIBNAME).scl : $(IORSRCS)
ifeq ($(IORSRCS),)
	echo "lib$(LIBNAME).scl is not needed for client-side C bindings."
else
	-rm -f $@
	echo '<?xml version="1.0" ?>' > $@
	echo '<scl>' >> $@
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; \
	   fi ; \
	  echo '  <library uri="'`pwd`/lib$(LIBNAME).la'" scope="'"$$scope"'" resolution="lazy" >' >> $@
	grep __set_epv $^ /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >>$@
	echo "  </library>" >>$@
	echo "</scl>" >>$@
endif

.SUFFIXES: .lo .chpl

.c.lo:
	babel-libtool --mode=compile --tag=CC $(CC) $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) -c -o $@ $<

.chpl.lo:
	$(CHPL) --savec $<.dir $< *Stub.h $(CHPL_HEADERS) --make true # don't use chpl to compile
	babel-libtool --mode=compile --tag=CC $(CC) \
            -I./$<.dir $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) \
            $(CHPL_FLAGS) -c -o $@ $<.dir/_main.c

clean :
	-rm -f $(PUREBABELGEN) babel-temp babel-stamp *.o *.lo

realclean : clean
	-rm -f lib$(LIBNAME).la lib$(LIBNAME).scl
	-rm -rf .libs

install : install-libs install-headers install-scl


install-libs : lib$(LIBNAME).la
	-mkdir -p $(LIBDIR)
	babel-libtool --mode=install install -c lib$(LIBNAME).la \
	  $(LIBDIR)/lib$(LIBNAME).la

install-scl : $(SCLFILE)
ifneq ($(IORSRCS),)
	-rm -f $(LIBDIR)/lib$(LIBNAME).scl
	-mkdir -p $(LIBDIR)
	echo '<?xml version="1.0" ?>' > $(LIBDIR)/lib$(LIBNAME).scl
	echo '<scl>' >> $(LIBDIR)/lib$(LIBNAME).scl
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; \
	   fi ; \
	  echo '  <library uri="'$(LIBDIR)/lib$(LIBNAME).la'" scope="'"$$scope"'" resolution="lazy" >' >> $(LIBDIR)/lib$(LIBNAME).scl
	grep __set_epv $^ /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >>$(LIBDIR)/lib$(LIBNAME).scl
	echo "  </library>" >>$(LIBDIR)/lib$(LIBNAME).scl
	echo "</scl>" >>$(LIBDIR)/lib$(LIBNAME).scl
endif

install-headers : $(IORHDRS) $(STUBHDRS)
	-mkdir -p $(INCLDIR)
	for i in $^ ; do \
	  babel-libtool --mode=install cp $$i $(INCLDIR)/$$i ; \
	done

.PHONY: all clean realclean install install-libs install-headers install-scl
""")

