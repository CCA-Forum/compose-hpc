#!/usr/bin/env python
# -*- python -*-
## @package chapel.chpl_backend
#
# Chapel glue code generator
#
# This is implemented as a converter that takes SIDL IR as input and
# creates semi-portable BRAID-IR as output. The glue code consists of
# two distinct parts, one being in Chapel (for things such as creating
# native Chapel objects out of Babel IOR objects) and the other being
# in C (for things such as low-level argument conversion and Babel
# virtual function calls).
#
# \authors <pre>
#
# Copyright (c) 2010-2013 Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
#
# Contributors/Acknowledgements:
#
# Summer interns at LLNL:
# * 2010, 2011 Shams Imam <shams@rice.edu>
#   contributed to argument conversions, r-array handling, exception
#   handling, the distributed array interface and the borrowed array
#   patch for the Chapel compiler
#
# LLNL-CODE-473891.
# All rights reserved.
#
# This file is part of BRAID ( http://compose-hpc.sourceforge.net/ ).
# Please read the COPYRIGHT file for Our Notice and for the BSD License.
#
# </pre>
#

import ior, ior_template, ir, os.path, re, sidl, sidlobjects, splicer
from utils import write_to, unzip
from patmat import *
from codegen import CFile, CCompoundStmt, c_gen
from sidl_symbols import visit_hierarchy
import chpl_conversions as conv
strip = conv.strip
from chpl_code import (ChapelFile, ChapelScope, chpl_gen, unscope, unscope_retval,
                  incoming, outgoing, gen_doc_comment, deref)
import backend
import chpl_makefile
import babel

chpl_data_var_template = '_babel_data_{arg_name}'
chpl_dom_var_template = '_babel_dom_{arg_name}'
chpl_local_var_template = '_babel_local_{arg_name}'
chpl_param_ex_name = '_ex'
extern_def_is_not_null = 'extern proc is_not_null(in aRef): bool;'
extern_def_set_to_null = 'extern proc set_to_null(inout aRef);'
chpl_base_interface = 'sidl.BaseInterface'
qual_id = babel.qual_id
retval_assignment = re.compile('^_retval\..+ =')

def c_struct(symbol_table, s):
    """
    lower scoped ids in item types
    FIXME: get rid of this function
    """
    def l((item, typ, name)):
        if ir.is_pointer_type(typ) and ir.is_scoped_id(typ[1]):
            return item, babel.lower_ir(symbol_table, typ[1]), name
        return item, typ, name
    struct, name, items, doc = s
    return struct, name, map(l, items), doc

def forward_decl(ir_struct):
    """
    \return a C-style forward declaration for a struct
    """
    return '%s %s;'%(ir_struct[0], ir_struct[1])

class GlueCodeGenerator(backend.GlueCodeGenerator):
    """
    This class provides the methods to transform SIDL to IR.
    """

    class ClassInfo(backend.GlueCodeGenerator.ClassInfo):
        """
        Holder object for the code generation scopes and other data
        during the traversal of the SIDL tree.
        """
        def __init__(self, class_object,
                     stub_parent=None,
                     skel_parent=None):

            assert(isinstance(class_object, object))
            self.co = class_object
            self.chpl_method_stub = ChapelFile(parent=stub_parent, relative_indent=4)
            self.chpl_skel = ChapelFile(parent=skel_parent, relative_indent=0)
            self.chpl_static_stub = ChapelFile(parent=stub_parent)
            self.chpl_static_skel = ChapelFile(parent=skel_parent)
            self.skel = CFile()
            self.epv = babel.EPV(class_object)
            self.ior = CFile('_'.join(class_object.qualified_name)+'_IOR')
            self.obj = None

    def __init__(self, config):
        super(GlueCodeGenerator, self).__init__(config)
        self.makefile = chpl_makefile

    @matcher(globals(), debug=False)
    def generate_glue_code(self, node, data, symbol_table, pkg_h=None):
        """
        Generate glue code for \c node .
        """
        def gen(node):
            return self.generate_glue_code(node, data, symbol_table, pkg_h)

        def generate_ext_stub(cls):
            """
            shared code for class/interface
            """
            # Qualified name (C Version)
            qname = '_'.join(symbol_table.prefix+[cls.name])

            if self.config.verbose:
                import sys
                mod_name = '.'.join(symbol_table.prefix[1:]+[cls.name])
                sys.stdout.write('\r'+' '*80)
                sys.stdout.write('\rgenerating glue code for %s'%mod_name)
                sys.stdout.flush()

            # Consolidate all methods, defined and inherited
            cls.scan_methods()

            # Initialize all class-specific code generation data structures
            chpl_stub = ChapelFile()
            # chpl_defs = ChapelScope(chpl_stub)
            ci = self.ClassInfo(cls, stub_parent=chpl_stub)

            if self.server:
                ci.impl = self.pkg_impl
                ci.impl.prefix=symbol_table.prefix

            chpl_stub.cstub.genh(ir.Import(qname+'_IOR'))
            chpl_stub.cstub.genh(ir.Import('sidl_header'))
            chpl_stub.cstub.genh(ir.Import('chpltypes'))
            chpl_stub.cstub.new_def(babel.externals(cls.get_scoped_id()))
            if self.server:
                chpl_stub.cstub.new_def(babel.builtin_stub_functions(cls.get_scoped_id()))

            has_contracts = ior_template.generateContractChecks(cls)
            self.gen_default_methods(cls, has_contracts, ci)
            #for m in ci.epv.methods: print m[3]
            #print qname, map(lambda x: x[2][1]+x[2][2], cls.all_methods)
            for method in cls.all_methods:
                (Method, Type, Name, Attrs, Args,
                 Except, From, Requires, Ensures, DocComment) = method
                ci.epv.add_method((method, Type, Name, Attrs,
                                   babel.drop_rarray_ext_args(Args),
                                   Except, From, Requires, Ensures, DocComment))

            builtins = [] if cls.is_interface() else babel.builtins
            # all the methods for which we would generate a server impl
            impl_methods = builtins+cls.get_methods()
            impl_methods_names = [sidl.method_method_name(m) for m in impl_methods]

            # client
            for method in builtins+cls.all_methods:
                has_impl = sidl.method_method_name(method) in impl_methods_names
                self.generate_client_method(symbol_table, method, ci, has_impl)

            if self.server:
                class_methods = filter(sidl.is_not_static, impl_methods)
                static_methods = filter(sidl.is_static, impl_methods)

                # Class
                ci.impl.new_def(gen_doc_comment(cls.doc_comment, chpl_stub)+
                                'class %s_Impl {'%qname)
                splicer = '.'.join(cls.qualified_name+['Impl'])
                ci.impl.new_def('// DO-NOT-DELETE splicer.begin(%s)'%splicer)
                ci.impl.new_def('// DO-NOT-DELETE splicer.end(%s)'%splicer)
                for method in class_methods:
                    self.generate_server_method(symbol_table, method, ci)

                ci.impl.new_def('} // class %s_Impl'%qname)
                ci.impl.new_def('')
                ci.impl.new_def('')

                # Static
                if static_methods:
                    ci.impl.new_def('// all static member functions of '+qname)
                    ci.impl.new_def(gen_doc_comment(cls.doc_comment, chpl_stub)+
                                    '// FIXME: chpl allows only one module per library //'+
                                    ' module %s_static_Impl {'%qname)

                    for method in static_methods:
                        self.generate_server_method(symbol_table, method, ci)

                    ci.impl.new_def('//} // module %s_static_Impl'%qname)
                    ci.impl.new_def('')
                    ci.impl.new_def('')


            # Chapel Stub (client-side Chapel bindings)
            self.generate_chpl_stub(chpl_stub, qname, ci)

            # Because of Chapel's implicit (filename-based) modules it
            # is important for the Chapel stub to be one file, but we
            # generate separate files for the cstubs
            self.pkg_chpl_stub.new_def(chpl_stub)

            # Stub (in C), the order of these definitions is somewhat sensitive
            cstub = chpl_stub.cstub
            cstub._name = qname+'_cStub'
            #cstub.genh_top(ir.Import(qname+'_IOR'))
            pkg_name = '_'.join(symbol_table.prefix)
            cstub.gen(ir.Import(pkg_name))
            cstub.gen(ir.Import(cstub._name))
            for code in cstub.optional:
                cstub.new_global_def(code)

            cstub.write()

            # IOR
            ci.ior.genh(ir.Import(pkg_name))
            if pkg_name <> 'sidl': 
                ci.ior.genh(ir.Import('sidl'))
            ci.ior.genh(ir.Import('chpltypes'))
            ior_template.generate_ior(ci, with_ior_c=self.server, _braid_config=self.config )
            ci.ior.write()

            # Skeleton
            if self.server:
                self.generate_skeleton(ci, qname)

            # Makefile
            self.classes.append(qname)


        if not symbol_table:
            raise Exception()

        with match(node):
            if (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                generate_ext_stub(sidlobjects.Class(symbol_table, node, self.class_attrs))

            elif (sidl.struct, (Name), Items, DocComment):
                # Generate Chapel stub
                struct_chpl_ior  = babel.lower_ir(symbol_table, node,
                                                  qualify_names=False, qualify_enums=False,
                                                  lower_scoped_ids=False, struct_suffix='')
                struct_chpl_chpl = conv.ir_type_to_chpl(struct_chpl_ior)
                struct_c_chpl    = babel.lower_ir(symbol_table, node, 
                                                  qualify_enums=False, raw_ior_arrays=True,
                                                  wrap_rarrays=True)
                self.pkg_chpl_stub.gen(ir.Type_decl(struct_c_chpl))
                self.pkg_chpl_stub.gen(ir.Type_decl(struct_chpl_chpl))

                struct_c_c       = babel.lower_ir(symbol_table, node, header=pkg_h)
                struct_chpl_c    = c_struct(symbol_table, conv.ir_type_to_chpl(struct_c_c))

                pkg_h.gen(ir.Type_decl(struct_c_c))
                pkg_h.new_header_def('#ifndef CHPL_GEN_CODE')
                pkg_h.genh(ir.Comment(
                        'Chapel will generate its own conflicting version of '+
                        "structs and enums since we can't use the extern "+
                        'keyword for them.'))
                pkg_h.gen(ir.Type_decl(struct_chpl_c))
                pkg_h.new_header_def('typedef struct %s %s;'%(struct_chpl_c[1], struct_chpl_c[1]))
                pkg_h.new_header_def('#else // CHPL_GEN_CODE')
                #pkg_h.new_header_def(forward_decl(struct_chpl_c))
                #pkg_h.new_header_def('#define %s __%s'%(struct_chpl_c[1], struct_chpl_c[1]))
                pkg_h.new_header_def('#endif // [not] CHPL_GEN_CODE')

                # record it for later, when the package is being finished
                # self.pkg_enums_and_structs.append(babel.struct_ior_names(node))
                # self.pkg_enums_and_structs.append(node)

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_ext_stub(sidlobjects.Interface(symbol_table, node, self.class_attrs))

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                enum_chpl_ior = babel.lower_ir(symbol_table, node, qualify_enums=False)
                enum_chpl     = conv.ir_type_to_chpl(enum_chpl_ior)
                self.pkg_chpl_stub.gen(ir.Type_decl(enum_chpl))
                enum_ior      = babel.lower_ir(symbol_table, node, header=pkg_h)
                pkg_h.gen(ir.Type_decl(enum_ior))

                # record it for later, when the package is being finished
                # self.pkg_enums_and_structs.append(node)

            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel stub
                qname = '_'.join(symbol_table.prefix+[Name])
                _, pkg_symbol_table = symbol_table[sidl.Scoped_id([], Name, '')]

                # header for package-wide definitions (enums, structs).
                pkg_h = CFile(qname)
                pkg_h.genh(ir.Import('sidl_header'))
                pkg_h.genh(ir.Import('chpltypes'))
                pkg_h.genh(ir.Comment('Braid package header file for'+
                                           '.'.join(symbol_table.prefix+[Name])))

                if self.in_package:
                    # nested modules are generated in-line
                    self.pkg_chpl_stub.new_def('module %s {'%Name)
                    self.generate_glue_code(UserTypes, data, pkg_symbol_table, pkg_h)
                    self.pkg_chpl_stub.new_def('}')
                else:
                    # server-side Chapel implementation template
                    if self.server: self.begin_impl(qname)

                    # new file for the toplevel package
                    self.pkg_chpl_stub = ChapelFile(qname, relative_indent=0)
                    self.pkg_chpl_stub.gen(ir.Import('sidl'))
                    self.pkg_chpl_stub.gen((ir.stmt, "extern proc generic_ptr(a:sidl__array):opaque"))
                    self.pkg_chpl_stub.gen((ir.stmt, "extern proc ptr_generic(a:opaque):sidl__array"))

                    # self.pkg_enums_and_structs = []
                    self.in_package = True

                    # recursion!
                    self.generate_glue_code(UserTypes, data, pkg_symbol_table, pkg_h)
                    #write_to(qname+'.chpl', str(self.pkg_chpl_stub))

                    # server-side Chapel implementation template
                    if self.server: self.end_impl(qname)

                    # Makefile
                    self.pkgs.append(qname)

                # write package header
                self.pkg_chpl_stub.write()
                pkg_h.write()

            elif (sidl.user_type, Attrs, Cipse):
                self.class_attrs = Attrs
                gen(Cipse)

            elif (sidl.file, Requires, Imports, UserTypes):
                self.in_package = False
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

    def struct_typedef(self, pkgname, s):
        if s[0] == sidl.enum: return ''
        return 'typedef {0} {1} _{1};\ntypedef _{1}* {1};'.format(
            s[0], pkgname+'_'+s[1][:-6])

    def class_header(self, qname, symbol_table, ci):
        header = CFile()
        pkgname = '_'.join(symbol_table.prefix)
        header._header = [
            '// Class header',
            '#include <stdint.h>',
            '#include <%s.h>' % pkgname,
            '#include <%s_IOR.h>'%qname,
            'typedef struct %s__object _%s__object;'%(qname, qname),
            'typedef _%s__object* %s__object;'%(qname, qname),
            '#ifndef included_sidl_BaseInterface_Stub_h',
            '#define included_sidl_BaseInterface_Stub_h',
            'typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;',
            'typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;',
            '#include <codelets.h>',
            '#endif'
            ]

        def gen_cast(_symtab, _ext, scope):
            """
            Chapel-specific up-cast macros
            """
            base = qual_id(scope)
            # Cast functions for the IOR
            header.genh(
   '#define _cast_{0}(ior,ex) ((struct {0}__object*)((*ior->d_epv->f__cast)(ior,"{1}",ex)))'
                       .format(base, qual_id(scope, '.')))
            header.genh('#define {1}_cast_{0}(ior) ((struct {1}__object*)'
                       '((struct sidl_BaseInterface__object*)ior)->d_object)'
                       .format(qname, base))

        extern_hier_visited = []
        for _, ext in ci.co.extends:
            visit_hierarchy(ext, gen_cast, symbol_table, extern_hier_visited)

        for _, impl in ci.co.implements:
            visit_hierarchy(impl, gen_cast, symbol_table, extern_hier_visited)

        return header

    chpl_conv_types = set([sidl.rarray, sidl.array, sidl.interface, sidl.class_])

    def babel_method_call(self, chpl_scope, symbol_table, cdecl, arguments, ci):
        # Build a burg tree for the function call
        def tmp(name):
            #return '_%s_ior_%s'%(ir.fn_decl_id(cdecl), name)
            return '_ior_%s'%name

        def incoming(((_, attrs, mode, typ, name), param_exp)):
            if mode <> sidl.out:
                param = ir.Deref(param_exp) if mode <> sidl.in_ else param_exp
                return conv.ir_to_burg(typ, 'chpl', symbol_table, False, tmp(name), param, [])
            else: return conv.outgoing_arg, (param_exp, tmp(name), None)

        def outgoing(((_, attrs, mode, typ, name), param_exp)):
            if mode <> sidl.in_:
                param = ir.Deref(param_exp) if param_exp <> '_retval' else param_exp
                return conv.ir_to_burg(typ, 'ior', symbol_table, False, param, tmp(name), [])
            else: return []

        def cons_with(f, l):
            l1 = [i for i in map(f, l) if i]
            if   len(l1) > 1: return reduce(lambda a, b: (conv.cons, a, b), l1)
            elif len(l1) > 0: return l1[0]
            else: import pdb; pdb.set_trace(); return conv.none,

        # Type conversion
        cdecl_type = ir.fn_decl_type(cdecl)
        crarg = (ir.arg, [], ir.out, cdecl_type, '_retval')
        if cdecl_type == ir.pt_void:
            rname, retval = [], []
        else:
            rname, retval = ['_retval'], [crarg]

        # Invoke the BURG tree pattern matcher
        if ir.static in ir.fn_decl_attrs(cdecl):
            method = (conv.nonvirtual_method, cdecl_type, ir.fn_decl_id(cdecl), ci)
        else:
            method = (conv.virtual_method, cdecl_type, ir.fn_decl_id(cdecl), ci)

        args = ir.fn_decl_args(cdecl)
        ins = cons_with(incoming, zip(args, arguments))
        outs = cons_with(outgoing, zip(args+retval, arguments+rname))
        assert(len(arguments) == len(args))
        burg_call = (conv.ior_call_assign, (conv.ior_call, method, ins), outs)
        #print burg_call
        conv.codegen(burg_call, conv.stmt, chpl_scope, chpl_scope.cstub)


    def babel_impl_call(self, chpl_scope, symbol_table, cdecl, arguments, ci):
        # Build a burg tree for the function call
        def tmp(name):
            #return '_%s_ior_%s'%(ir.fn_decl_id(cdecl), name)
            return '_chpl_%s'%name

        def incoming(((_, attrs, mode, typ, name), param_exp)):
            if mode <> sidl.out:
                param = ir.Deref(param_exp) if mode <> sidl.in_ else param_exp
                return conv.ir_to_burg(typ, 'ior', symbol_table, False, tmp(name), param, [])
            else: return conv.outgoing_arg, (param_exp, tmp(name), None)

        def outgoing(((_, attrs, mode, typ, name), param_exp)):
            if mode <> sidl.in_:
                param = ir.Deref(param_exp) if param_exp <> '_retval' else param_exp
                return conv.ir_to_burg(typ, 'chpl', symbol_table, False, param, tmp(name), [])
            else: return []

        def cons_with(f, l):
            l1 = [i for i in map(f, l) if i]
            if   len(l1) > 1: return reduce(lambda a, b: (conv.cons, a, b), l1)
            elif len(l1) > 0: return l1[0]
            else: import pdb; pdb.set_trace(); return conv.none,

        # Type conversion
        cdecl_type = ir.fn_decl_type(cdecl)
        crarg = (ir.arg, [], ir.out, cdecl_type, '_retval')
        if cdecl_type == ir.pt_void:
            rname, retval = [], []
        else:
            rname, retval = ['_retval'], [crarg]

        # Invoke the BURG tree pattern matcher
        if ir.static in ir.fn_decl_attrs(cdecl):
            method = (conv.nonvirtual_method, cdecl_type, ir.fn_decl_id(cdecl), ci)
        else:
            method = (conv.virtual_method, cdecl_type, ir.fn_decl_id(cdecl), ci)
        args = ir.fn_decl_args(cdecl)
        ins = cons_with(incoming, zip(args, arguments))
        outs = cons_with(outgoing, zip(args+retval, arguments+rname))
        assert(len(arguments) == len(args))
        burg_call = (conv.chpl_call_assign, (conv.chpl_call, method, ins), outs)
        #print burg_call
        conv.codegen(burg_call, conv.stmt, chpl_scope, chpl_scope.cstub)



    @matcher(globals(), debug=False)
    def generate_client_method(self, symbol_table, method, ci, has_impl):
        """
        Generate client code for a method interface.
        \param method        s-expression of the method's SIDL declaration
        \param symbol_table  the symbol table of the SIDL file
        \param ci            a ClassInfo object

        The code generation proceeds in two big steps, one is here and
        creates a Chapel stub in Chapel that does as much of the
        argument -> IOR conversion as possible, but we also generate a
        secon stub in C. It might be possible to move the bulk of
        argument conversions into small C functions or macros for each
        type, but at least the virtual call can only be implemented in
        C (or in a macro).
        """
        def lower_rarray_args((arg, attrs, mode, typ, name)):
            if typ[0] == sidl.rarray:
                return arg, attrs, mode, conv.get_array_type(typ)[1], name
            return arg, attrs, mode, typ, name

        def structs_identical((s1, n1, items1, d1), (s2, n2, items2, d2)):
            '''
            compare two structs for structural equivalence (aka duck typing)
            '''
            def struct_item_identical(((i1, typ1, n1), (i2, typ2, n2))):
                if ir.is_struct(typ1) and ir.is_struct(typ2):
                    return structs_identical(typ1, typ2)
                return typ1 == typ2
            
            return reduce(lambda x,y: x and y,
                          map(struct_item_identical, zip(items1, items2)))


        # Chapel stub
        (Method, Type, (MName,  Name, Extension), Attrs, Args,
         Except, From, Requires, Ensures, DocComment) = method

        #ior_type = babel.lower_ir(symbol_table, Type, lower_scoped_ids=False)
        #ior_args = babel.lower_ir(symbol_table, babel.drop_rarray_ext_args(Args),
        #                          lower_scoped_ids=False)

        cdecl_type = babel.lower_ir(symbol_table, Type)
        
        #cdecl_self = babel.lower_ir(symbol_table, ci.co.get_qualified_data())
        cdecl_args = babel.lower_ir(symbol_table, babel.drop_rarray_ext_args(Args))
        #chpl_stub_args = babel.epv_args(Attrs, cdecl_args, symbol_table, ci.epv.name)

        #map(lambda arg: conv.sidl_arg_to_ir(symbol_table, arg), ior_args)

        chpl_args = babel.lower_ir(symbol_table, babel.drop_rarray_ext_args(Args), 
                                   lower_scoped_ids=False,
                                   qualify_names=False, qualify_enums=True,
                                   struct_suffix='')
        chpl_type = babel.lower_ir(symbol_table, Type, lower_scoped_ids=False,
                                   qualify_names=False, qualify_enums=True)
        chpl_type = conv.ir_type_to_chpl(chpl_type)

        abstract = member_chk(sidl.abstract, Attrs)
        #final = member_chk(sidl.final, Attrs)
        static = member_chk(sidl.static, Attrs)

        #attrs = []
        if abstract:
            # we still need to output a stub for an abstract function,
            # since it might me a virtual function call through an
            # abstract interface
            pass

        if static:
            #attrs.append(ir.static)
            chpl_scope = ci.chpl_static_stub
            selfarg = []
            m = ci.epv.find_static_method(Name+Extension)
        else:
            chpl_scope = ci.chpl_method_stub
            selfarg = ['self']
            m = ci.epv.find_method(Name+Extension)

        # # this is an ugly hack to force generate_method_stub to to wrap the
        # # self argument with a call to upcast()
        # if ci.co.is_interface():
        #     docast = [ir.pure]
        # else: docast = []

        chpl_scope.prefix=symbol_table.prefix
        body = ChapelScope(chpl_scope)

        if selfarg: body.gen((ir.stmt, 'var self = this'))

        stub_ex = '_ex'
        body.new_def(extern_def_is_not_null)
        body.new_def(extern_def_set_to_null)
        body.gen(ir.Stmt(ir.Call("set_to_null", [stub_ex])))

        # return value type conversion -- treat it as an out argument
        #srarg = (sidl.arg, [], sidl.out, Type, '_retval')
        #rarg  = (ir.arg, [], sidl.out, cdecl_type, '_retval')
        #crarg = (ir.arg, [], ir.out, cdecl_type, '_retval') #conv.sidl_arg_to_ir(symbol_table, rarg)
        #_,_,_,ctype,_ = crarg
        args = selfarg+map(lambda arg: ir.arg_id(arg), cdecl_args)+[stub_ex]

        self.babel_method_call(body, symbol_table, m, args, ci)

        def ext_to_chpl_classtype(typ):
            """
            deal with the dual representation of Extensibles as raw sidl
            pointers and Chapel objects
            """
            if babel.is_obj_type(symbol_table, typ):
                _, prefix, name, ext = typ
                return ir.Typedef_type('.'.join(prefix+[name]))
            ##if name =='_ior__retval':print typ, name
            #if ir.is_pointer_type(typ) and ir.is_struct(typ[1]):
            #    #if name == '_ior__retval': name = '_retval'
            #    #if name == '_retval': name = '_ior_retval'
            #    if typ[1][1][0] == sidl.scoped_id:
            #        _, prefix, tname, ext = typ[1][1]
            #        if ext == '__object' and name == '_retval':
            #            #name <> '_ex' and name[:4] <> '_ior_': 
            #            return ir.Typedef_type('.'.join(list(prefix)+[tname])), name

            return typ


        # FIXME, this is ugly!
        if Type <> sidl.void:
            if ((Type[0] == sidl.scoped_id and
                 symbol_table[Type][1][0] in self.chpl_conv_types)
                or Type[0] in self.chpl_conv_types):
                body.genh(ir.Stmt(ir.Var_decl(ext_to_chpl_classtype(chpl_type), '_retval')))
            else:
                #if ir.is_struct(chpl_type):
                #    body.genh(ir.Var_decl(chpl_type, '_ior__retval'))
                body.genh(ir.Stmt(ir.Var_decl(ext_to_chpl_classtype(chpl_type), '_retval')))
                ## this needs to go ...
                #assigns_retval = False
                #for stmt in body._defs:
                #    if re.match(retval_assignment, stmt):
                #        assigns_retval = True
                #        break
                #if not assigns_retval:
                #    body.gen(ir.Copy('_retval', '_ior__retval'))

            body.genh(ir.Comment(str(Type)))
            body.gen(ir.Stmt(ir.Return('_retval')))

        # Add the exception to the chapel method signature
        chpl_stub_args = chpl_args + [
            ir.Arg([], ir.out, (ir.typedef_type, chpl_base_interface), '_ex')]

        defn = (ir.fn_defn, [], chpl_type,
                Name + Extension, chpl_stub_args,
                [str(body)],
                DocComment)

        chpl_scope.gen(defn)
        return

    def vcall(self, chpl_scope, classname, name, args, ci):
        params = 'in obj, out ex'
        chpl_scope.genh((ir.stmt, 'extern proc %s_%s_cStub(%s)'%(classname, name, params)))
        chpl_scope.gen(ir.Stmt(ir.Call('%s_%s_cStub'%(classname, name), args)))

    def generate_chpl_stub(self, chpl_stub, qname, ci):
        """
        Chapel Stub (client-side Chapel bindings)

        Generate a Stub in Chapel.

        Chapel supports C structs via the extern keyword,
        but they must be typedef'ed in a header file that
        must be passed to the chpl compiler.
        """
        symbol_table = ci.epv.symbol_table
        cls = ci.co

        # Qualified name including Chapel modules
        mod_qname = '.'.join(symbol_table.prefix[1:]+[qname])
        mod_name = '.'.join(symbol_table.prefix[1:]+[cls.name])

        header = self.class_header(qname, symbol_table, ci)
        write_to(qname+'_Stub.h', header.dot_h(qname+'_Stub.h'))
        if self.server:
            chpl_stub.gen(ir.Import('%s_Impl'%'_'.join(symbol_table.prefix)))
        extrns = ChapelScope(chpl_stub, relative_indent=0)

        def gen_extern_casts(_symtab, _ext, baseclass):
            base = qual_id(baseclass)
            mod_base = '.'.join(baseclass[1]+[base])
            ex = 'out ex: sidl_BaseInterface__object'
            extrns.new_def('extern proc _cast_{0}(in ior: {1}__object, {2}): {3}__object;'
                           .format(base, mod_qname, ex, mod_base))
            extrns.new_def('extern proc {3}_cast_{1}(in ior: {0}__object): {2}__object;'
                           .format(mod_base, qname, mod_qname, base))

        parent_classes = []
        extern_hier_visited = []
        for _, ext in cls.extends:
            visit_hierarchy(ext, gen_extern_casts, symbol_table,
                            extern_hier_visited)
            parent_classes += babel.strip_common(symbol_table.prefix, ext[1])

        parent_interfaces = []
        for _, impl in cls.implements:
            visit_hierarchy(impl, gen_extern_casts, symbol_table,
                            extern_hier_visited)
            parent_interfaces += babel.strip_common(symbol_table.prefix, impl[1])

        inherits = ''
        interfaces = ''
        if parent_classes:
            inherits = ' /*' + ': ' + ', '.join(parent_classes) + '*/ '
        if parent_interfaces:
            interfaces = ' /*' + ', '.join(parent_interfaces) + '*/ '

        # extern declaration for the IOR
        chpl_stub.new_def('extern record %s__object {'%qname)
        chpl_stub.new_def('};')

        chpl_stub.new_def(extrns)
        chpl_stub.new_def('extern proc %s__createObject('%qname+
                             'd_data: int, '+
                             'out ex: sidl_BaseInterface__object)'+
                             ': %s__object;'%mod_qname)
        name = chpl_gen(cls.name)

        chpl_class = ChapelScope(chpl_stub)
        chpl_static_helper = ChapelScope(chpl_stub, relative_indent=4)

        self_field_name = 'self_' + name
        # Generate create and wrap methods for classes to init/wrap the IOR

        # The field to point to the IOR
        chpl_class.new_def('var ' + self_field_name + ': %s__object;' % mod_qname)

        common_head = [
            '  ' + extern_def_is_not_null,
            '  ' + extern_def_set_to_null,
            '  var ex: sidl_BaseInterface__object;',
            '  set_to_null(ex);'
        ]
        common_tail = [
            '  if (is_not_null(ex)) {',
            '     {arg_name} = new {base_ex}(ex);'.format(arg_name=chpl_param_ex_name, base_ex=chpl_base_interface) ,
            '  }'
        ]

        # The create() method to create a new IOR instance
        create_body = ChapelScope(chpl_class)
        create_body.gen(common_head)
        create_body.new_def('  this.' + self_field_name + ' = %s__createObject(0, ex);' % qname)
        self.vcall(create_body, qname, 'addRef', ['this.' + self_field_name, 'ex'], ci)
        create_body.gen(common_tail)
        wrapped_ex_arg = ir.Arg([], ir.out, (ir.typedef_type, chpl_base_interface), chpl_param_ex_name)
        if not cls.is_interface():
            # Interfaces instances cannot be created!
            chpl_class.gen(
                (ir.fn_defn, [], ir.pt_void,
                 'init_' + name,
                 [wrapped_ex_arg],
                 [str(create_body)], 'Pseudo-Constructor to initialize the IOR object'))
            # Create a static function to create the object using create()
            wrap_static_defn = (ir.fn_defn, [], mod_name,
                'create', #_' + name,
                [wrapped_ex_arg],
                [
                    '  var inst = new %s();' % mod_name,
                    '  inst.init_%s(%s);' % (name, wrapped_ex_arg[4]),
                    '  return inst;'
                ],
                'Static helper function to create instance using create()')
            chpl_gen(wrap_static_defn, chpl_static_helper)

        # This wrap() method to copy the refernce to an existing IOR
        wrap_body = []
        wrap_body.extend(common_head)
        wrap_body.append('  this.' + self_field_name + ' = obj;')
        wrap_body.extend(common_tail)
        wrapped_obj_arg = ir.Arg([], ir.in_, babel.ir_object_type(symbol_table.prefix, name), 'obj')
        chpl_gen(
            (ir.fn_defn, [], ir.pt_void,
             'wrap',
             [wrapped_obj_arg, wrapped_ex_arg],
             wrap_body,
             'Pseudo-Constructor for wrapping an existing object'), chpl_class)

        # Create a static function to create the object using wrap()
        wrap_static_defn = (ir.fn_defn, [], mod_name,
            'wrap_' + name,
            [wrapped_obj_arg, wrapped_ex_arg],
            [
                '  var inst = new %s();' % mod_name,
                '  inst.wrap(%s, %s);' % (wrapped_obj_arg[4], wrapped_ex_arg[4]),
                '  return inst;'
            ],
            'Static helper function to create instance using wrap()')
        chpl_gen(wrap_static_defn, chpl_static_helper)

        # Provide a destructor for the class
        destructor_body = ChapelScope(chpl_class)
        destructor_body.new_def('var ex: sidl_BaseInterface__object;')
        self.vcall(destructor_body, qname, 'deleteRef', ['this.' + self_field_name, 'ex'], ci)
        if not cls.is_interface():
            # Interfaces have no destructor
            self.vcall(destructor_body, qname, '_dtor', ['this.' + self_field_name, 'ex'], ci)
        chpl_class.gen(
            (ir.fn_defn, [], ir.pt_void, '~'+chpl_gen(name), [],
             [str(destructor_body)], 'Destructor'))

        def gen_self_cast():
            chpl_gen(
                (ir.fn_defn, [], (ir.typedef_type, '%s__object' % mod_qname),
                 'as_'+qname, [],
                 ['return %s;' % self_field_name],
                 'return the current IOR pointer'), chpl_class)

        def gen_cast(_symtab, _ext, base):
            qbase = qual_id(base)
            mod_base = '.'.join(base[1]+[qbase])
            chpl_gen(
                (ir.fn_defn, [], (ir.typedef_type, '%s__object' % mod_base),
                 'as_'+qbase, [],
                 ['var ex: sidl_BaseInterface__object;',
                  ('return _cast_%s(this.' + self_field_name + ', ex);') % qbase],
                 'Create a up-casted version of the IOR pointer for\n'
                 'use with the alternate constructor'), chpl_class)

        gen_self_cast()
        casts_generated = [symbol_table.prefix+[name]]
        for _, ext in cls.extends:
            visit_hierarchy(ext, gen_cast, symbol_table, casts_generated)
        for _, impl in cls.implements:
            visit_hierarchy(impl, gen_cast, symbol_table, casts_generated)

        # chpl_class.new_def(chpl_defs.get_defs())

        # chpl_stub.new_def(chpl_defs.get_decls())
        chpl_stub.new_def('// All the static methods of class '+name)
        chpl_stub.new_def('module %s_static {'%name)
        chpl_stub.new_def(ci.chpl_static_stub.get_defs())
        chpl_stub.new_def(chpl_static_helper)
        chpl_stub.new_def('}')
        chpl_stub.new_def('')
        chpl_stub.new_def(gen_doc_comment(cls.doc_comment, chpl_stub)+
                          'class %s %s %s {'%(name,inherits,interfaces))
        chpl_stub.new_def(chpl_class)
        chpl_stub.new_def(ci.chpl_method_stub)
        chpl_stub.new_def('}')


    def generate_skeleton(self, ci, qname):
        """
        Chapel Skeleton (client-side Chapel bindings)

        Generate a Skeleton in Chapel.
        """
        symbol_table = ci.epv.symbol_table
        cls = ci.co


        # Skeleton (in Chapel)
        self.pkg_chpl_skel.gen(ir.Import('.'.join(symbol_table.prefix)))

        self.pkg_chpl_skel.new_def('use sidl;')
        # objname = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name]) + '_Impl'

        self.pkg_chpl_skel.new_def('extern record %s__object { var d_data: opaque; };'
                                   %qname)#,objname))
        self.pkg_chpl_skel.new_def('extern proc %s__createObject('%qname+
                             'd_data: int, '+
                             'out ex: sidl_BaseInterface__object)'+
                             ': %s__object;'%qname)
        self.pkg_chpl_skel.new_def(ci.chpl_skel)


        # Skeleton (in C)
        cskel = ci.chpl_skel.cstub
        cskel._name = qname+'_Skel'
        cskel.gen(ir.Import('stdint'))
        cskel.gen(ir.Import('stdio'))
        cskel.gen(ir.Import(cskel._name))
        cskel.gen(ir.Import(qname+'_IOR'))
        cskel.gen(ir.Fn_defn([], ir.pt_void, qname+'__call_load', [],
                               [ir.Comment("FIXME: [ir.Stmt(ir.Call('_load', []))")], ''))

        # set_epv ... Setup the entry-point vectors (EPV)s
        #
        # there are 2*3 types of EPVs:
        #    epv: regular methods
        #    sepv: static methods
        #    pre_(s)epv: pre-hooks
        #    post_(s)epv: post-hooks
        epv_t = ci.epv.get_ir()
        sepv_t = ci.epv.get_sepv_ir()
        pre_epv_t   = ci.epv.get_pre_epv_ir()
        pre_sepv_t  = ci.epv.get_pre_sepv_ir()
        post_epv_t  = ci.epv.get_post_epv_ir()
        post_sepv_t = ci.epv.get_post_sepv_ir()
        cskel.gen(ir.Fn_decl([], ir.pt_void, 'ctor', [], ''))
        cskel.gen(ir.Fn_decl([], ir.pt_void, 'dtor', [], ''))

        epv_init  = []
        sepv_init = []
        for m in babel.builtins+cls.get_methods():
            fname =  m[2][1] + m[2][2]
            attrs = sidl.method_method_attrs(m)
            static = member_chk(sidl.static, attrs)
            def entry(stmts, epv_t, table, field, pointer):
                stmts.append(ir.Set_struct_item_stmt(epv_t, ir.Deref(table), field, pointer))

            if static: entry(sepv_init, sepv_t, 'sepv', 'f_'+fname, '%s_%s_skel'%(qname, fname))
            else:      entry(epv_init,   epv_t,  'epv', 'f_'+fname, '%s_%s_skel'%(qname, fname))

            builtin_names = ['_ctor', '_ctor2', '_dtor']
            with_hooks = member_chk(ir.hooks, attrs)
            if fname not in builtin_names and with_hooks:
                if static: entry(sepv_init, pre_sepv_t,  'pre_sepv',  'f_%s_pre'%fname,  'NULL')
                else:      entry(epv_init,  pre_epv_t,   'pre_epv',   'f_%s_pre'%fname,  'NULL')
                if static: entry(sepv_init, post_sepv_t, 'post_sepv', 'f_%s_post'%fname, 'NULL')
                else:      entry(epv_init,  post_epv_t,  'post_epv',  'f_%s_post'%fname, 'NULL')

        pkgname = '_'.join(ci.epv.symbol_table.prefix)

        dummyargv = '''
  const char* argv[] = {
    babel_program_name,
    "-nl", /* number of locales */
    "",
    "-v", /* verbose chapel runtime */
    NULL
  };
  argv[2] = getenv("SLURM_NTASKS");
  if (argv[2] == NULL) {
    fprintf(stdout, "**ERROR: please set the SLURM_NTASKS environment variable\\n"
                    "         to the desired number of Chapel locales.");
    argv[2] = "0";
  }
  int ignored = setenv("GASNET_BACKTRACE", "1", 1);
'''
        cskel.genh(ir.Import('stdlib'))
        cskel.pre_def('extern int chpl_init_library(int argc, char* argv[]);')
        cskel.pre_def('// You can set this to argv[0] in main() to get better debugging output')
        cskel.pre_def('char* __attribute__((weak)) babel_program_name = "BRAID_LIBRARY";')
        init_code = [dummyargv, 'int locale_id = chpl_init_library(4, argv)']
        init_code = map(lambda x: (ir.stmt, x), init_code)
        epv_init.extend(init_code)
        sepv_init.extend(init_code)

        cskel.gen(ir.Fn_defn(
            [], ir.pt_void, qname+'__set_epv',
            [ir.Arg([], ir.out, epv_t, 'epv'),
             ir.Arg([], ir.out, pre_epv_t, 'pre_epv'),
             ir.Arg([], ir.out, post_epv_t, 'post_epv')],
            epv_init, ''))

        if sepv_t:
            cskel.gen(ir.Fn_defn(
                    [], ir.pt_void, qname+'__set_sepv',
                    [ir.Arg([], ir.out, sepv_t, 'sepv'),
                     ir.Arg([], ir.out, pre_sepv_t, 'pre_sepv'),
                     ir.Arg([], ir.out, post_sepv_t, 'post_sepv')],
                    sepv_init, ''))

        # C Skel
        for code in cskel.optional:
            cskel.new_global_def(code)
        cskel.write()

    def begin_impl(self, qname):
        """
        Chapel Impl (server-side Chapel implementation template)

        Start generating a module_Impl.chpl file in Chapel.
        """
        # new file for the toplevel package
        self.pkg_chpl_skel = ChapelFile(qname+'_Skel')
        #self.pkg_chpl_skel.main_area.new_def('proc __defeat_dce(){\n')

        # new file for the user implementation
        self.pkg_impl = ChapelFile(qname+'_Impl')
        self.pkg_impl.gen(ir.Import('sidl'))
        self.pkg_impl.new_def(extern_def_set_to_null)
        self.pkg_impl.new_def('// DO-NOT-DELETE splicer.begin(%s.Impl)'%qname)
        self.pkg_impl.new_def('// DO-NOT-DELETE splicer.end(%s.Impl)'%qname)
        self.pkg_impl.new_def('')


    def end_impl(self, qname):
        """
        Chapel Impl (server-side Chapel implementation template)

        Finish generating the module_Impl.chpl file in Chapel.
        """
        # write the Chapel skeleton to disk
        self.pkg_chpl_skel.write()

        # deal with the impl file
        # if self.pkg_enums_and_structs:
        self.pkg_impl._header.append(chpl_gen(ir.Import(qname)))

        impl = qname+'_Impl.chpl'

        # Preserve code written by the user
        if os.path.isfile(impl):
            # FIXME: this is a possible race condition, we should
            # use a single file handle instead
            splicers = splicer.record(impl)
            lines = str(self.pkg_impl).split('\n')
            write_to(impl, splicer.apply_all(impl, lines, splicers))
        else:
            write_to(impl, str(self.pkg_impl))

    @matcher(globals(), debug=False)
    def generate_server_method(self, symbol_table, method, ci):
        """
        Generate server code for a method interface.  This function
        generates a C-callable skeleton for the method and generates a
        Skeleton of Chapel code complete with splicer blocks for the
        user to fill in.

        \param method        s-expression of the method's SIDL declaration
        \param symbol_table  the symbol table of the SIDL file
        \param ci            a ClassInfo object
        """

        def convert_arg((arg, attrs, mode, typ, name)):
            """
            Extract name and generate argument conversions
            """
            iorname = name
            return iorname, (arg, attrs, mode, typ, name)


        # Chapel skeleton
        (Method, Type, (MName,  Name, Extension), Attrs, Args,
         Except, From, Requires, Ensures, DocComment) = method

         #ior_args = drop_rarray_ext_args(Args)

#        ci.epv.add_method((Method, Type, (MName,  Name, Extension), Attrs, ior_args,
#                           Except, From, Requires, Ensures, DocComment))

        abstract = member_chk(sidl.abstract, Attrs)
        static = member_chk(sidl.static, Attrs)
        #final = member_chk(sidl.static, Attrs)

        if abstract:
            # nothing to be done for an abstract function
            return

        decls = []
        pre_call = []
        call_args = []
        post_call = []
        ior_args = babel.lower_ir(symbol_table, Args, lower_scoped_ids=False)
        ior_type = babel.lower_ir(symbol_table, Type, lower_scoped_ids=False)
        return_stmt = []
        skel = ci.chpl_skel
        opt = skel.cstub.optional
        qname = '_'.join(ci.co.qualified_name+[Name+Extension])
        callee = qname+'_impl'

        if static:
            #chpl_scope = ci.chpl_static_stub
            selfarg = []
            m = ci.epv.find_static_method(Name+Extension)
        else:
            #chpl_scope = ci.chpl_method_stub
            selfarg = ['self']
            m = ci.epv.find_method(Name+Extension)

        body = ChapelScope(ci.chpl_skel)
        body.prefix=symbol_table.prefix
        body.cstub = CCompoundStmt(ci.chpl_skel)
        body.cstub.optional = ci.chpl_skel.cstub.optional


        # Add the exception to the chapel method signature
        chpl_skel_args = ior_args + [
            ir.Arg([], ir.out, babel.ir_baseinterface_type(), '_ex')]

        args = selfarg+map(lambda arg: ir.arg_id(arg), ior_args)+['_ex']
        self.babel_impl_call(body, symbol_table, m, args, ci)

        skeldefn = (ir.fn_defn, [], ior_type,
                    qname+'_skel', 
                    babel.epv_args(Attrs, Args, ci.epv.symbol_table, ci.epv.name),
                    [str(body.cstub)],
                    DocComment)

        def skel_args((arg, attr, mode, typ, name)):
            if typ[0] == sidl.array:
                # lower array args
                import pdb; pdb.set_trace()
                return arg, attr, mode, ir.pt_void, name
            elif mode == ir.in_ and ir.is_typedef_type(typ) and (
                # complex is always passed as a pointer since chpl 1.5
                typ[1] == '_complex64' or
                typ[1] == '_complex128'):
                return arg, attr, mode, ir.Pointer_type(typ), name
            else: return arg, attr, mode, typ, name

        def unscoped_args((arg, attr, mode, typ, name)):
            if ir.is_enum(typ):
                return (arg, attr, mode,
                        (ir.enum, unscope(ci.epv.symbol_table, typ[1]), typ[2], typ[3]),
                        name)
            return arg, attr, mode, typ, name

        #ex_arg = [ir.Arg([], ir.inout, babel.ir_baseinterface_type(), '_ex')]
        this_arg = [] if static else [ir.Arg([], ir.in_, ir.void_ptr, '_self')]
        impl_args = map(skel_args, chpl_skel_args)#+ex_arg
        #impldecl = (ir.fn_decl, [], ior_type, callee, this_arg+impl_args, DocComment)
        splicer = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name, Name])
        impldefn = (ir.fn_defn, ['export '+callee],
                    ior_type, #unscope_retval(ci.epv.symbol_table, ior_type),
                    Name,
                    impl_args, #map(unscoped_args, impl_args),
                    ['set_to_null(_ex);',
                     '// DO-NOT-DELETE splicer.begin(%s)'%splicer,
                     '// DO-NOT-DELETE splicer.end(%s)'%splicer],
                    DocComment)

        c_gen(skeldefn, ci.chpl_skel.cstub)
        #c_gen(impldecl, ci.chpl_skel.cstub)
        chpl_gen(impldefn, ci.impl)
