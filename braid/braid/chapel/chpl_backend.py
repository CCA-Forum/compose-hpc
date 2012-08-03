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
# Copyright (c) 2010, 2011, 2012 Lawrence Livermore National Security, LLC.
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

import ior_template, ir, os.path, sidl, sidlobjects, splicer
from utils import write_to, unzip
from patmat import *
from codegen import CFile, c_gen
from sidl_symbols import visit_hierarchy
import chpl_conversions as conv
from chpl_code import (ChapelFile, ChapelScope, chpl_gen, unscope, unscope_retval,
                  incoming, outgoing, gen_doc_comment, strip, deref)
import backend
import chpl_makefile
import babel

chpl_data_var_template = '_babel_data_{arg_name}'
chpl_dom_var_template = '_babel_dom_{arg_name}'
chpl_local_var_template = '_babel_local_{arg_name}'
chpl_param_ex_name = '_babel_param_ex'
extern_def_is_not_null = 'extern proc IS_NOT_NULL(in aRef): bool;'
extern_def_set_to_null = 'extern proc SET_TO_NULL(inout aRef);'
chpl_base_interface = 'BaseInterface'
qual_id = babel.qual_id

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
    def generate_glue_code(self, node, data, symbol_table):
        """
        Generate glue code for \c node .
        """
        def gen(node): 
            return self.generate_glue_code(node, data, symbol_table)

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

            chpl_stub.cstub.genh(ir.Import(qname+'_IOR'))
            chpl_stub.cstub.genh(ir.Import('sidl_header'))
            chpl_stub.cstub.genh(ir.Import('chpl_sidl_array'))
            chpl_stub.cstub.genh(ir.Import('chpltypes'))
            chpl_stub.cstub.new_def(babel.externals(cls.get_scoped_id()))
            if self.server: 
                chpl_stub.cstub.new_def(babel.builtin_stub_functions(cls.get_scoped_id()))

            has_contracts = ior_template.generateContractChecks(cls)
            self.gen_default_methods(cls, has_contracts, ci)

            #print qname, map(lambda x: x[2][1]+x[2][2], cls.all_methods)
            for method in cls.all_methods:
                (Method, Type, Name, Attrs, Args, 
                 Except, From, Requires, Ensures, DocComment) = method
                ci.epv.add_method((method, Type, Name, Attrs, 
                                   babel.drop_rarray_ext_args(Args),
                                   Except, From, Requires, Ensures, DocComment))

            # all the methods for which we would generate a server impl
            impl_methods = babel.builtins+cls.get_methods()
            impl_methods_names = [sidl.method_method_name(m) for m in impl_methods]

            # client
            for method in cls.all_methods:
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
            cstub.genh_top(ir.Import(qname+'_IOR'))
            cstub.gen(ir.Import(cstub._name))
            for code in cstub.optional:
                cstub.new_global_def(code)

            pkg_name = '_'.join(symbol_table.prefix)
            cstub.gen(ir.Import(pkg_name))
            cstub.write()

            # IOR
            ci.ior.genh(ir.Import(pkg_name))
            ci.ior.genh(ir.Import('sidl'))
            ci.ior.genh(ir.Import('chpl_sidl_array'))
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
                self.pkg_chpl_stub.gen(ir.Type_decl(babel.lower_ir(symbol_table, node, struct_suffix='')))

                # record it for later, when the package is being finished
                self.pkg_enums_and_structs.append(babel.struct_ior_names(node))

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_ext_stub(sidlobjects.Interface(symbol_table, node, self.class_attrs))

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_stub.gen(ir.Type_decl(node))

                # record it for later, when the package is being finished
                self.pkg_enums_and_structs.append(node)
                
            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel stub
                qname = '_'.join(symbol_table.prefix+[Name])
                _, pkg_symbol_table = symbol_table[sidl.Scoped_id([], Name, '')]

                if self.in_package:
                    # nested modules are generated in-line
                    self.pkg_chpl_stub.new_def('module %s {'%Name)
                    self.generate_glue_code(UserTypes, data, pkg_symbol_table)
                    self.pkg_chpl_stub.new_def('}')
                else:
                    # server-side Chapel implementation template
                    if self.server: self.begin_impl(qname)

                    # new file for the toplevel package
                    self.pkg_chpl_stub = ChapelFile(relative_indent=0)
                    self.pkg_enums_and_structs = []
                    self.in_package = True
                    
                    # recursion!
                    self.generate_glue_code(UserTypes, data, pkg_symbol_table)
                    write_to(qname+'.chpl', str(self.pkg_chpl_stub))

                    # server-side Chapel implementation template
                    if self.server: self.end_impl(qname)
     
                    # Makefile
                    self.pkgs.append(qname)

                pkg_h = CFile(qname)

                pkg_h.genh(ir.Import('sidl_header'))
                pkg_h.genh(ir.Import('chpltypes'))
                for es in self.pkg_enums_and_structs:
                    es_ior = babel.lower_ir(pkg_symbol_table, es, header=pkg_h, qualify_names=True)
                    pkg_h.gen(ir.Type_decl(es_ior))
                    # generate also the chapel version of the struct, if different
                    if es[0] == sidl.struct:
                        es_chpl = conv.ir_type_to_chpl(es_ior)
                        if es_chpl <> es_ior: 
                            pkg_h.new_header_def('#ifndef CHPL_GEN_CODE')
                            pkg_h.genh(ir.Comment(
                                    'Chapel will generate its own conflicting version of'+
                                    "structs and enums since we can't use the extern "+
                                    'keyword for them.'))
                            pkg_h.gen(ir.Type_decl(es_chpl))
                            pkg_h.new_header_def('#else // CHPL_GEN_CODE')
                            pkg_h.new_header_def(forward_decl(es_chpl))
                            pkg_h.new_header_def('#endif // [not] CHPL_GEN_CODE')

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
            '#define IS_NULL(aPtr)     ((aPtr) == 0)',
            '#define IS_NOT_NULL(aPtr) ((aPtr) != 0)',
            '#define SET_TO_NULL(aPtr) ((*aPtr) = 0)',
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

        def convert_arg((arg, attrs, mode, typ, name)):
            """
            Extract name and generate argument conversions
            """
            iorname = name
            iortype = typ
            if babel.is_obj_type(symbol_table, typ):
                iortype = babel.ior_type(symbol_table, typ)
                if mode <> sidl.out:
                    iorname = name + '.self_' + typ[2]

                if mode <> sidl.in_:
                    iorname = '_IOR_' + name
                    
                    pre_call.append(ir.Stmt(ir.Var_decl(iortype, iorname)))
                    
                    # wrap the C type in a native Chapel object
                    chpl_class_name = typ[2]
                    mod_chpl_class_name = '.'.join(typ[1]+[chpl_class_name])
                    conv = ir.Call(qual_id(typ, '.') + '_static.wrap_' + chpl_class_name, 
                                   [iorname, chpl_param_ex_name])
                    
                    if name == '_retval':
                        post_call.append(ir.Stmt(ir.Var_decl(
                                    (ir.typedef_type, mod_chpl_class_name), name)))
                    post_call.append(ir.Stmt(ir.Assignment(name, conv)))
                    
                    if name == '_retval':
                        return_expr.append(name)
            
            elif babel.is_struct_type(symbol_table, typ):
                iortype = babel.lower_ir(*symbol_table[typ])
                if (mode == sidl.in_):
                    iortype = ir.Pointer_type(iortype)

            elif typ[0] == sidl.scoped_id:
                # Other Symbol
                iortype = symbol_table[typ][1]

            elif typ == sidl.void:
                iortype = ir.pt_void

            elif typ == sidl.opaque:
                iortype = ir.Pointer_type(ir.pt_void)

            elif typ == (sidl.array, [], [], []): 
                # This comment is there so the Chapel code generator
                # won't recognize the type as an array. Here is the
                # only place where we actually want a C struct type.
                iortype = ir.Pointer_type(ir.Struct('sidl__array /* IOR */', [], ''))
                if name == '_retval':
                    iortype = ir.Pointer_type(ir.pt_void)

            elif typ[0] == sidl.array: # Scalar_type, Dimension, Orientation
                if typ[1][0] == ir.scoped_id:
                    t = 'BaseInterface'
                else:
                    t = typ[1][1]
                iortype = ir.Pointer_type(ir.Struct('sidl_%s__array /* IOR */'%t, [], ''))
                if mode <> sidl.out:
                    iorname = name+'.ior'

                if mode <> sidl.in_:
                    iorname = '_IOR_' + name
                    # wrap the C type in a native Chapel object
                    pre_call.append(ir.Stmt(ir.Var_decl(iortype, iorname)))
                    if mode == sidl.inout:
                        pre_call.append(ir.Stmt(ir.Assignment(iorname, name+'.ior')))

                    conv = (ir.new, 'sidl.Array', [typ[1], 'sidl_%s__array'%t, iorname])
                    
                    if name == '_retval':
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
                iortype = ir.Typedef_type('sidl_%s__array'%typ[1][1])
                arg_name = name# convert_el_res[0]
                chpl_data_var_name = chpl_data_var_template.format(arg_name=arg_name)
                chpl_dom_var_name = chpl_dom_var_template.format(arg_name=arg_name)
                chpl_local_var_name = chpl_local_var_template.format(arg_name=arg_name)
                
                # sanity check on input array: ensure domain is rectangular
                pre_call.append(ir.Stmt(ir.Call('performSanityCheck',
                    [chpl_dom_var_name, '"{arg_name}"'.format(arg_name=arg_name)])))
                # ensure we are working with a 'local' array
                # FIXME Hack to define a variable without explicitly specifying type
                # should we change the IR to support this?
                pre_call.append(ir.Stmt(ir.Assignment('var ' + chpl_data_var_name,
                    ir.Call("getOpaqueData", [ir.Call(arg_name, [chpl_dom_var_name + ".low"])])))
                )
                pre_call.append(ir.Stmt(ir.Assignment('var ' + chpl_local_var_name,
                    ir.Call("ensureLocalArray", [arg_name, chpl_data_var_name])))
                )
                
                if original_mode <> sidl.in_:
                    # emit code to copy back elements into non-local array
                    chpl_wrapper_ior_name = "_babel_wrapped_local_{arg}".format(arg=arg_name)
                    chpl_wrapper_sarray_name = "_babel_wrapped_local_{arg}_sarray".format(arg=arg_name)
                    chpl_wrapper_barray_name = "_babel_wrapped_local_{arg}_barray".format(arg=arg_name)
                    post_call.append(ir.Stmt(ir.Assignment('var ' + chpl_wrapper_sarray_name,
                        ir.Call("new Array", ["{arg}.eltType".format(arg=arg_name),
                                 iortype[1], chpl_wrapper_ior_name]))))
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
            var {a}rank = _babel_dom_{arg}.rank: int(32);

            var {a}lus = computeLowerUpperAndStride(_babel_local_{arg});
            var {a}lower = {a}lus(0): int(32);
            var {a}upper = {a}lus(1): int(32);
            var {a}stride = {a}lus(2): int(32);
            
            var _babel_wrapped_local_{arg}: {iortype} = {iortype}_borrow(
                {stype}_ptr(_babel_local_{arg}(_babel_local_{arg}.domain.low)),
                //_babel_local_{arg}(_babel_local_{arg}.domain.low),
                {a}rank,
                {a}lower[1],
                {a}upper[1],
                {a}stride[1])""".format(a='_babel_%s_'%arg_name,
                                         arg=arg_name,
                                         iortype=iortype[1],
                                         stype=typ[1][1]))
                
                pre_call.append(sidl_wrapping)
                post_call.append((ir.stmt, '//sidl__array_deleteRef((struct sidl__array*)a_tmp)'))

                # reference the lowest element of the array using the domain
                #call_expr_str = chpl_local_var_name + '(' + chpl_local_var_name + '.domain.low' + ')'
                #return (call_expr_str, convert_el_res[1])
                return '_babel_wrapped_local_'+arg_name, (arg, attrs, mode, iortype, name)
                
            return iorname, (arg, attrs, mode, iortype, name)

        def obj_by_value((arg, attrs, mode, typ, name)):
            if babel.is_obj_type(symbol_table, typ):
                return (arg, attrs, sidl.in_, typ, name)
            else:
                return (arg, attrs, mode, typ, name)

        # Chapel stub
        (Method, Type, (MName,  Name, Extension), Attrs, Args,
         Except, From, Requires, Ensures, DocComment) = method

        ior_type = babel.lower_ir(symbol_table, Type, struct_suffix='', lower_scoped_ids=False)
        ior_args = babel.lower_ir(symbol_table, babel.drop_rarray_ext_args(Args), 
                                  struct_suffix='', lower_scoped_ids=False)

        chpl_args = []
        chpl_args.extend(babel.lower_ir(symbol_table, Args, struct_suffix='', 
                                        lower_scoped_ids=False, qualify_names=False))

        abstract = member_chk(sidl.abstract, Attrs)
        #final = member_chk(sidl.final, Attrs)
        static = member_chk(sidl.static, Attrs)

        attrs = []
        if abstract:
            # we still need to output a stub for an abstract function,
            # since it might me a virtual function call through an
            # abstract interface
            pass
        if static: attrs.append(ir.static)

        # this is an ugly hack to force generate_method_stub to to wrap the
        # self argument with a call to upcast()
        if ci.co.is_interface():
            docast = [ir.pure]
        else: docast = []

        pre_call = []
        post_call = []
        return_expr = []
        return_stmt = []

        pre_call.append(extern_def_is_not_null)
        pre_call.append(extern_def_set_to_null)
        pre_call.append(ir.Stmt(ir.Var_decl(babel.ir_exception_type(), '_ex')))
        pre_call.append(ir.Stmt(ir.Call("SET_TO_NULL", ['_ex'])))
        
        post_call.append(ir.Stmt(ir.If(
            ir.Call("IS_NOT_NULL", ['_ex']),
            [
                ir.Stmt(ir.Assignment(chpl_param_ex_name,
                                   ir.Call("new " + chpl_base_interface, ['_ex'])))
            ]
        )))
        
        call_args, cdecl_args = unzip(map(convert_arg, ior_args))
        
        # return value type conversion -- treat it as an out argument
        _, (_,_,_,iortype,_) = convert_arg((ir.arg, [], ir.out, ior_type, '_retval'))
        if iortype[0] == ir.enum: 
            chpl_type = babel.lower_ir(symbol_table, Type, struct_suffix='', 
                                       lower_scoped_ids=False, qualify_names=False)
        else: chpl_type = iortype

        cdecl_args = babel.stub_args(attrs, cdecl_args, symbol_table, ci.epv.name, docast)
        cdecl = ir.Fn_decl(attrs, iortype, Name + Extension, cdecl_args, DocComment)

        if static:
            call_self = []
        else:
            call_self = ['this.self_' + ci.epv.name]
                

        call_args = call_self + call_args + ['_ex']
        # Add the exception to the chapel method signature
        chpl_args.append(ir.Arg([], ir.out, (ir.typedef_type, chpl_base_interface), chpl_param_ex_name))
        if self.server and has_impl:
            # if we are generating server code we can take a shortcut
            # and directly invoke the implementation
            modname = '_'.join(ci.co.symbol_table.prefix+['Impl'])
            if not static:
                qname = '_'.join(ci.co.qualified_name+['Impl'])
                # FIXME!
            callee = '.'.join([modname, ir.fn_decl_id(cdecl)])
        else:
            callee = babel.build_function_call(ci, cdecl, static)

        stubcall = ir.Call(callee, call_args)
        if Type == sidl.void:
            Type = ir.pt_void
            call = [ir.Stmt(stubcall)]
        else:
            if return_expr or post_call:
                rvar = '_IOR__retval'
                if not return_expr:                                
                    if iortype[0] == ir.struct:
                        iortype = babel.lower_ir(*symbol_table[Type], struct_suffix='')

                    pre_call.append(ir.Stmt(ir.Var_decl(chpl_type, rvar)))
                    rx = rvar
                else:
                    rx = return_expr[0]

                if babel.is_struct_type(symbol_table, Type):
                    # use rvar as an additional OUT argument instead
                    # of a return value because Chapel cannot deal
                    # with return-by-value classes and every struct
                    # must be either a struct (value) or a record (reference)
                    call = [ir.Stmt(ir.Call(callee, call_args+[rvar]))]
                else:
                    call = [ir.Stmt(ir.Assignment(rvar, stubcall))]
                return_stmt = [ir.Stmt(ir.Return(rx))]
            else:
                call = [ir.Stmt(ir.Return(stubcall))]

        defn = (ir.fn_defn, [],
                babel.lower_ir(symbol_table, Type, struct_suffix='', 
                               lower_scoped_ids=False, qualify_names=False),
                Name + Extension, chpl_args,
                pre_call+call+post_call+return_stmt,
                DocComment)

        if static:
            ci.chpl_static_stub.prefix=symbol_table.prefix
            chpl_gen(defn, ci.chpl_static_stub)
        else:
            ci.chpl_method_stub.prefix=symbol_table.prefix
            chpl_gen(defn, ci.chpl_method_stub)


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
        chpl_stub.gen(ir.Import('sidl'))
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
            '  SET_TO_NULL(ex);'
        ]
        common_tail = [
            babel.vcall('addRef', ['this.' + self_field_name, 'ex'], ci),
            '  if (IS_NOT_NULL(ex)) {',
            '     {arg_name} = new {base_ex}(ex);'.format(arg_name=chpl_param_ex_name, base_ex=chpl_base_interface) ,
            '  }'
        ]

        # The create() method to create a new IOR instance
        create_body = []
        create_body.extend(common_head)
        create_body.append('  this.' + self_field_name + ' = %s__createObject(0, ex);' % qname)
        create_body.extend(common_tail)
        wrapped_ex_arg = ir.Arg([], ir.out, (ir.typedef_type, chpl_base_interface), chpl_param_ex_name)
        if not cls.is_interface():
            # Interfaces instances cannot be created!
            chpl_gen(
                (ir.fn_defn, [], ir.pt_void,
                 'init_' + name,
                 [wrapped_ex_arg],
                 create_body, 'Pseudo-Constructor to initialize the IOR object'), chpl_class)
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
        wrapped_obj_arg = ir.Arg([], ir.in_, babel.ir_object_type([], qname), 'obj')
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
        destructor_body = []
        destructor_body.append('var ex: sidl_BaseInterface__object;')
        destructor_body.append(babel.vcall('deleteRef', ['this.' + self_field_name, 'ex'], ci))
        if not cls.is_interface():
            # Interfaces have no destructor
            destructor_body.append(babel.vcall('_dtor', ['this.' + self_field_name, 'ex'], ci))
        chpl_gen(
            (ir.fn_defn, [], ir.pt_void, '~'+chpl_gen(name), [],
             destructor_body, 'Destructor'), chpl_class)

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
  char* argv[] = { 
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
        # These are now called by chpl_init_library -> chpl_gen_init
        #cskel.pre_def('extern void chpl__init_chpl__Program(int, const char*);')
        #cskel.pre_def('extern void chpl__init_%s_Impl(int, const char*);'%pkgname)
        init_code = [dummyargv,
                 'int locale_id = chpl_init_library(4, argv)',
        #         'chpl__init_chpl__Program(__LINE__, __FILE__)',
        #         'chpl__init_%s_Impl(__LINE__, __FILE__)'%pkgname
                     ]
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
        if self.pkg_enums_and_structs:
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


    def generate_server_es_defs(self, pkg_symbol_table, es, qname):
        """
        Write the package-wide definitions (enums, structs).
        """
        pkg_chpl = ChapelFile(qname)
        pkg_h = CFile(qname)
        pkg_h.genh(ir.Import('sidl_header'))
        for es in self.pkg_enums_and_structs:
            es_ior = babel.lower_ir(pkg_symbol_table, es, header=pkg_h, qualify_names=True)
            es_chpl = es_ior
            if es[0] == sidl.struct:
                es_ior = conv.ir_type_to_chpl(es_ior)
                es_chpl = conv.ir_type_to_chpl(es)

            pkg_h.gen(ir.Type_decl(es_ior))
            pkg_chpl.gen(ir.Type_decl(es_chpl))

        pkg_h.write()
        pkg_chpl.write()
    
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
        ctype = babel.lower_ir(symbol_table, Type, lower_scoped_ids=False)
        return_stmt = []
        skel = ci.chpl_skel
        opt = skel.cstub.optional
        qname = '_'.join(ci.co.qualified_name+[Name])
        callee = qname+'_impl'
     
        # Argument conversions
        # ---------------------

        # self
        this_arg = [] if static else [ir.Arg([], ir.in_, ir.void_ptr, '_this')]
     
        # IN
        map(lambda (arg, attr, mode, typ, name):
              conv.codegen((strip(typ), deref(mode, typ, name)), ('chpl', strip(typ)),
                           pre_call, skel, '_CHPL_'+name, typ),
            filter(incoming, ior_args))
     
        # OUT
        map(lambda (arg, attr, mode, typ, name):
              conv.codegen((('chpl', strip(typ)), '_CHPL_'+name), strip(typ),
                           post_call, skel, '(*%s)'%name, typ),
            filter(outgoing, ior_args))

        # RETURN value type conversion -- treated just like an OUT argument
        rarg = (ir.arg, [], ir.out, ctype, '_retval')
        conv.codegen((('chpl', strip(ctype)), '_CHPL__retval'), strip(ctype),
                     post_call, skel, '_retval', ctype)
        chpl_rarg = conv.ir_arg_to_chpl(rarg)
        _,_,_,chpltype,_ = chpl_rarg
        if Type <> sidl.void:
            decls.append(ir.Stmt(ir.Var_decl(ctype, '_retval')))

        # def pointerize_struct((arg, attr, mode, typ, name)):
        #   # FIXME: this is borked.. instead we should remove this
        #   # _and_ the code in codegenerator that strips the
        #   # pointer_type again
        #   if typ[0] == ir.struct:
        #         return (arg, attr, mode, (ir.pointer_type, typ), name)
        #   else: return (arg, attr, mode, typ, name)

        # chpl_args = map(pointerize_struct, map(conv.ir_arg_to_chpl, ior_args))
        chpl_args = map(conv.ir_arg_to_chpl, ior_args)

     
        # Proxy declarations / revised names of call arguments
        is_retval = True
        for (_,attrs,mode,chpl_t,name), (_,_,_,c_t,_) \
                in zip([chpl_rarg]+chpl_args, [rarg]+ior_args):

            if chpl_t <> c_t:
                is_struct = False
                proxy_t = chpl_t
                if c_t[0] == ir.pointer_type and c_t[1][0] == ir.struct:
                    # inefficient!!!
                    opt.add(str(c_gen(ir.Type_decl(chpl_t[1]))))
                    c_t = c_t[1]
                    is_struct = True
                    proxy_t = chpl_t[1]
     
                # FIXME see comment in chpl_to_ior
                name = '_CHPL_'+name
                decls.append(ir.Stmt(ir.Var_decl(proxy_t, name)))
                if (mode <> sidl.in_ or is_struct 
                    # TODO this should be handled by a conversion rule
                    or (mode == sidl.in_ and (
                            c_t == ir.pt_fcomplex or 
                            c_t == ir.pt_dcomplex))):
                    name = ir.Pointer_expr(name)
     
            if name == 'self' and member_chk(ir.pure, attrs):
                # part of the hack for self dereferencing
                upcast = ('({0}*)(((struct sidl_BaseInterface__object*)self)->d_object)'
                          .format(c_gen(c_t[1])))
                call_args.append(upcast)
            else:
                if is_retval: is_retval = False
                else:         call_args.append(name)

        call_args.append('_ex')

        if not static:
            call_args = ['self->d_data']+call_args

        # The actual function call
        if Type == sidl.void:
            Type = ir.pt_void
            call = [ir.Stmt(ir.Call(callee, call_args))]
        else:
            if post_call:
                call = [ir.Stmt(ir.Assignment('_CHPL__retval', ir.Call(callee, call_args)))]
                return_stmt = [ir.Stmt(ir.Return('_retval'))]
            else:
                call = [ir.Stmt(ir.Return(ir.Call(callee, call_args)))]

        #TODO: ior_args = drop_rarray_ext_args(Args)

        skeldefn = (ir.fn_defn, [], ctype, qname+'_skel',
                    babel.epv_args(Attrs, Args, ci.epv.symbol_table, ci.epv.name),
                    decls+pre_call+call+post_call+return_stmt,
                    DocComment)

        def skel_args((arg, attr, mode, typ, name)):
            if typ[0] == sidl.array:
                # lower array args
                return arg, attr, mode, ir.pt_void, name
            elif mode == ir.in_ and typ[0] == ir.typedef_type and (
                # complex is always passed as a pointer since chpl 1.5
                typ[1] == '_complex64' or
                typ[1] == '_complex128'):
                return arg, attr, mode, ir.Pointer_type(typ), name
            else: return arg, attr, mode, typ, name

        def unscoped_args((arg, attr, mode, typ, name)):
            if typ[0] == ir.enum:
                return (arg, attrs, ir.inout, 
                        (ir.enum, unscope(ci.epv.symbol_table, typ[1]), typ[2], typ[3]),
                        name)
            return arg, attr, mode, typ, name

        ex_arg = [ir.Arg([], ir.inout, babel.ir_baseinterface_type(), '_ex')]
        impl_args = this_arg+map(skel_args, chpl_args)+ex_arg
        impldecl = (ir.fn_decl, [], chpltype, callee, impl_args, DocComment)
        splicer = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name, Name])
        impldefn = (ir.fn_defn, ['export '+callee], 
                    unscope_retval(ci.epv.symbol_table, chpltype), 
                    Name,
                    map(unscoped_args, impl_args),
                    ['SET_TO_NULL(_ex);',
                     '// DO-NOT-DELETE splicer.begin(%s)'%splicer,
                     '// DO-NOT-DELETE splicer.end(%s)'%splicer],
                    DocComment)

        c_gen(skeldefn, ci.chpl_skel.cstub)
        c_gen(impldecl, ci.chpl_skel.cstub)
        chpl_gen(impldefn, ci.impl)




