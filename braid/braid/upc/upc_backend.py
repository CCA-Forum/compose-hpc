#!/usr/bin/env python
# -*- python -*-
## @package upc.backend
#
# UPC glue code generator
#
# \authors <pre>
#
# Copyright (c) 2012 Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
# 
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
from codegen import (CFile)
from sidl_symbols import visit_hierarchy
import babel, backend, sidl, upc_code, upc_makefile

qual_id = babel.qual_id
upc_gen = upc_code.upc_gen

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
            qname = '_'.join(class_object.qualified_name)
            self.co = class_object
            self.stub = upc_code.UPCFile(qname+'_Stub')
            self.skel = upc_code.UPCFile(qname+'_Skel')
            self.epv = babel.EPV(class_object)
            self.ior = upc_code.UPCFile(qname+'_IOR')
            self.obj = None


    def __init__(self, config):
        super(GlueCodeGenerator, self).__init__(config)
        self.makefile = upc_makefile
        self.exts = []

    @matcher(globals(), debug=False)
    def generate_glue_code(self, node, data, symbol_table):
        """
        Generate glue code for \c node .        
        """
        def gen(node): return self.generate_glue_code(node, data, symbol_table)

        def generate_ext_stub(cls):
            """
            shared code for class/interface
            """
            # Qualified name (C Version)
            qname = '_'.join(symbol_table.prefix+[cls.name])
            self.exts.append(qname)

            if self.config.verbose:
                import sys
                mod_name = '.'.join(symbol_table.prefix[1:]+[cls.name])
                sys.stdout.write('\r'+' '*80)
                sys.stdout.write('\rgenerating glue code for %s'%mod_name)
                sys.stdout.flush()

            # Consolidate all methods, defined and inherited
            cls.scan_methods()
                
            # chpl_defs = ChapelScope(chpl_stub)
            ci = self.ClassInfo(cls)

            # if self.server:
            #     ci.impl = self.pkg_impl

            ci.stub.new_def(babel.externals(cls.get_scoped_id()))
            ci.stub.new_def(babel.builtin_stub_functions(cls.get_scoped_id()))
                
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

            #     # Class
            #     ci.impl.new_def(gen_doc_comment(cls.doc_comment, chpl_stub)+
            #                     'class %s_Impl {'%qname)
            #     splicer = '.'.join(cls.qualified_name+['Impl'])
            #     ci.impl.new_def('// DO-NOT-DELETE splicer.begin(%s)'%splicer)
            #     ci.impl.new_def('// DO-NOT-DELETE splicer.end(%s)'%splicer)
            #     for method in class_methods:  
            #         self.generate_server_method(symbol_table, method, ci)

            #     ci.impl.new_def('} // class %s_Impl'%qname)
            #     ci.impl.new_def('')
            #     ci.impl.new_def('')

            #     # Static
            #     if static_methods:
            #         ci.impl.new_def('// all static member functions of '+qname)
            #         ci.impl.new_def(gen_doc_comment(cls.doc_comment, chpl_stub)+
            #                         '// FIXME: chpl allows only one module per library //'+
            #                         ' module %s_static_Impl {'%qname)

            #         for method in static_methods:
            #             self.generate_server_method(symbol_table, method, ci)

            #         ci.impl.new_def('//} // module %s_static_Impl'%qname)
            #         ci.impl.new_def('')
            #         ci.impl.new_def('')


            # # Chapel Stub (client-side Chapel bindings)
            # self.generate_chpl_stub(chpl_stub, qname, ci)
            
            # # Because of Chapel's implicit (filename-based) modules it
            # # is important for the Chapel stub to be one file, but we
            # # generate separate files for the cstubs
            # self.pkg_chpl_stub.new_def(chpl_stub)

            # Stub (in C), the order of these definitions is somewhat sensitive
            ci.stub.genh_top(ir.Import(qname+'_IOR'))
            ci.stub.gen(ir.Import(ci.stub._name))

            pkg_name = '_'.join(symbol_table.prefix)
            ci.stub.gen(ir.Import(pkg_name))
            ci.stub.write()

            # IOR
            ior_template.generate_ior(ci, with_ior_c=self.server, _braid_config=self.config )
            ci.ior.write()

            # Skeleton
            if self.server:
                self.generate_skeleton(ci, qname)

            # Convenience header
            ext_h = CFile(qname)
            ext_h.genh(ir.Import(qname+'_IOR'))
            ext_h.genh(ir.Import(qname+'_Stub'))
            ext_h.write()

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
                # self.pkg_chpl_stub.gen(ir.Type_decl(lower_ir(symbol_table, node, struct_suffix='')))

                # record it for later, when the package is being finished
                self.pkg_enums_and_structs.append(struct_ior_names(node))

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_ext_stub(sidlobjects.Interface(symbol_table, node, self.class_attrs))

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                # self.pkg_chpl_stub.gen(ir.Type_decl(node))

                # record it for later, when the package is being finished
                self.pkg_enums_and_structs.append(node)
                
            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel stub
                qname = '_'.join(symbol_table.prefix+[Name])
                _, pkg_symbol_table = symbol_table[sidl.Scoped_id([], Name, '')]

                if self.in_package:
                    # nested modules are generated in-line
                    # self.pkg_chpl_stub.new_def('module %s {'%Name)
                    self.generate_glue_code(UserTypes, data, pkg_symbol_table)
                    # self.pkg_chpl_stub.new_def('}')
                else:
                    # server-side Chapel implementation template
                    if self.server: self.begin_impl(qname)

                    # new file for the toplevel package
                    # self.pkg_chpl_stub = ChapelFile(relative_indent=0)
                    self.pkg_enums_and_structs = []
                    self.in_package = True
                    
                    # recursion!
                    self.generate_glue_code(UserTypes, data, pkg_symbol_table)
                    # write_to(qname+'.chpl', str(self.pkg_chpl_stub))

                    # server-side Chapel implementation template
                    if self.server: self.end_impl(qname)
     
                    # Makefile
                    self.pkgs.append(qname)

                pkg_h = CFile(qname)
                pkg_h = pkg_h
                pkg_h.genh(ir.Import('sidl_header'))
                for es in self.pkg_enums_and_structs:
                    es_ior = babel.lower_ir(pkg_symbol_table, es, header=pkg_h, qualify_names=True)
                    pkg_h.gen(ir.Type_decl(es_ior))

                for ext in self.exts:
                    pkg_h.genh(ir.Import(ext))

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

  
    @matcher(globals(), debug=False)
    def generate_client_method(self, symbol_table, method, ci, has_impl):
        """
        Generate client code for a method interface.
        \param method        s-expression of the method's SIDL declaration
        \param symbol_table  the symbol table of the SIDL file
        \param ci            a ClassInfo object

        This is currently a no-op, since the UPC calling convention is
        identical to the IOR.
        """
        (Method, Type, (MName,  Name, Extension), Attrs, Args,
         Except, From, Requires, Ensures, DocComment) = method

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

        ior_type = babel.lower_ir(symbol_table, Type)
        ior_args = babel.epv_args(Attrs, Args, symbol_table, ci.epv.name)
        call_args = map(lambda arg: ir.arg_id(arg), ior_args)
        cdecl = ir.Fn_decl(attrs, ior_type, Name + Extension, ior_args, DocComment)
        qname = '_'.join(ci.co.qualified_name+[Name]) + Extension

        if self.server and has_impl:
            # if we are generating server code we can take a shortcut
            # and directly invoke the implementation
            modname = '_'.join(ci.co.symbol_table.prefix+['Impl'])
            if not static:
                qname = '_'.join(ci.co.qualified_name+['Impl'])
                # FIXME!
            callee = '_'.join([modname, ir.fn_decl_id(cdecl)])
        else:
            callee = babel.build_function_call(ci, cdecl, static)

        if Type == sidl.void:
            call = [ir.Stmt(ir.Call(callee, call_args))]
        else:
            call = [ir.Stmt(ir.Return(ir.Call(callee, call_args)))]

        cdecl = ir.Fn_decl(attrs, ior_type, qname, ior_args, DocComment)
        cdefn = ir.Fn_defn(attrs, ior_type, qname, ior_args, call, DocComment)

        if static:
            # TODO: [performance] we would only need to put the
            # _externals variable into the _Stub.c, not necessarily
            # all the function definitions
            ci.stub.gen(cdecl)
            ci.stub.new_def('#pragma weak '+qname)
            ci.stub.gen(cdefn)
        else:
            # FIXME: can't UPC handle the inline keyword??
            ci.stub.new_header_def('static inline')
            ci.stub.genh(cdefn)
            # ci.stub.gen(cdecl)
            # ci.stub.gen(cdefn)


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
        objname = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name]) + '_Impl'

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
        for m in builtins+cls.get_methods():
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
        pkg_h.genh(ir.Import('sidlType'))
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
            # lower array args
            if typ[0] == sidl.array:
                return arg, attr, mode, ir.pt_void, name
            # complex is always passed as a pointer since chpl 1.5
            elif mode == ir.in_ and typ[0] == ir.typedef_type and (
                typ[1] == '_complex64' or
                typ[1] == '_complex128'):
                return arg, attr, mode, ir.Pointer_type(typ), name
            else: return arg, attr, mode, typ, name

        ex_arg = [ir.Arg([], ir.inout, babel.ir_baseinterface_type(), '_ex')]
        impl_args = this_arg+map(skel_args, chpl_args)+ex_arg
        impldecl = (ir.fn_decl, [], chpltype, callee, impl_args, DocComment)
        splicer = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name, Name])
        impldefn = (ir.fn_defn, ['export '+callee], 
                    chpltype, Name, impl_args,
                    ['SET_TO_NULL(_ex);',
                     '// DO-NOT-DELETE splicer.begin(%s)'%splicer,
                     '// DO-NOT-DELETE splicer.end(%s)'%splicer],
                    DocComment)

        c_gen(skeldefn, ci.stub)
        c_gen(impldecl, ci.stub)
        upc_gen(impldefn, ci.impl)




