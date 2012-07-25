#!/usr/bin/env python
# -*- python -*-
## @package backend
#
# glue code generator interface
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
from codegen import c_gen
import babel, ir, ior_template, sidl

class GlueCodeGenerator(object):
    """
    This class provides the methods to transform SIDL to IR.
    """
    
    class ClassInfo(object):
        """
        Holder object for the code generation scopes and other data
        during the traversal of the SIDL tree.
        """
        def __init__(self, class_object):

            assert(isinstance(class_object, object))
            self.co = class_object

    def __init__(self, config):
        """
        Create a new code generator
        \param config.gen_hooks       generate pre-/post-method hooks
        \param config.gen_contracts   generate contract enforcement code
        \param config.make_prefix     prefix for babel.make
        \param config.verbose         if \c True, print extra messages
        """
        self.config = config
        self.classes = []
        self.pkgs = []
        self.sidl_files = ''
        self.makefile = None

    def generate_makefile(self):
        """
        Generate a GNUmakefile for all the entities
        generated by this GlueCodeGenerator.
        """
        self.makefile.gen_gnumakefile(self.sidl_files)

    def generate_bindings(self, filename, sidl_ast, symbol_table, with_server):
        """
        Generate client or server code. Operates in two passes:

        \li create symbol table
        \li do the work

        Server bindings always include client bindings, in case server
        code wants to call itself.

        \param filename        full path to the SIDL file
        \param sidl_ast        s-expression of the SIDL data
        \param symbol_table    symbol table for the SIDL data
        \param with_server     if \c true, generate server bindings, too
        """
        self.server = with_server
        self.sidl_files += ' '+filename
        try:
            self.generate_glue_code(sidl_ast, None, symbol_table)
            if self.config.verbose:
                import sys
                sys.stdout.write("%s\r"%(' '*80))

            if self.server:
                self.makefile.gen_server(self.sidl_files, self.classes, self.pkgs,
                                         self.config.make_prefix)
            else:
                self.makefile.gen_client(self.sidl_files, self.classes,
                                         self.config.make_prefix)

        except:
            # Invoke the post-mortem debugger
            import pdb, sys
            print sys.exc_info()
            pdb.post_mortem()


    def generate_glue_code(self, node, data, symbol_table):
        """
        Generate glue code for \c node .
        """
        assert(False)

    def gen_default_methods(self, cls, has_contracts, ci):
        """
        Generate default Babel object methods such as _cast() Also
        generates other IOR data structures such as the _object and
        the controls and statistics struct.
        """

        def unscope((struct, scoped_id, items, doc)):
            return struct, c_gen(scoped_id), items, doc

        def builtin(t, name, args):
            ci.epv.add_method(
                sidl.Method(t, sidl.Method_name(name, ''), [],
                            args, [], [], [], [], 
                            'Implicit built-in method: '+name))

        def static_builtin(t, name, args):
            ci.epv.add_method(
                sidl.Method(t, sidl.Method_name(name, ''), [sidl.static],
                            args, [], [], [], [], 
                            'Implicit built-in method: '+name))

        def inarg(t, name):
            return sidl.Arg([], sidl.in_, t, name)

        # Implicit Built-in methods
        builtin(sidl.pt_opaque, '_cast',
                [inarg(sidl.pt_string, 'name')])

        builtin(sidl.void, '_delete', [])

        builtin(sidl.void, '_exec', [
                inarg(sidl.pt_string, 'methodName'),
                inarg(babel.object_type(['sidl', 'rmi'], 'Call'), 'inArgs'),
                inarg(babel.object_type(['sidl', 'rmi'], 'Return'), 'outArgs')])

        builtin(sidl.pt_string, '_getURL', [])
        builtin(sidl.void, '_raddRef', [])
        builtin(sidl.pt_bool, '_isRemote', [])
        builtin(sidl.void, '_set_hooks', 
                [inarg(sidl.pt_bool, 'enable')])
        builtin(sidl.void, '_set_contracts', [
                inarg(sidl.pt_bool, 'enable'),
                inarg(sidl.pt_string, 'enfFilename'),
                inarg(sidl.pt_bool, 'resetCounters')],
                )
        builtin(sidl.void, '_dump_stats', 
                [inarg(sidl.pt_string, 'filename'),
                 inarg(sidl.pt_string, 'prefix')])
        if not cls.is_interface():
            builtin(sidl.void, '_ctor', [])
            builtin(sidl.void, '_ctor2',
                    [(sidl.arg, [], sidl.in_, ir.void_ptr, 'private_data')])
            builtin(sidl.void, '_dtor', [])
            builtin(sidl.void, '_load', [])

        static_builtin(sidl.void, '_set_hooks_static', 
                [inarg(sidl.pt_bool, 'enable')])
        static_builtin(sidl.void, '_set_contracts_static', [
                inarg(sidl.pt_bool, 'enable'),
                inarg(sidl.pt_string, 'enfFilename'),
                inarg(sidl.pt_bool, 'resetCounters')],
                )
        static_builtin(sidl.void, '_dump_stats_static', 
                [inarg(sidl.pt_string, 'filename'),
                 inarg(sidl.pt_string, 'prefix')])


        prefix = ci.epv.symbol_table.prefix
        # cstats
        cstats = []
        contract_cstats = []
        ci.methodcstats = []
        num_methods = cls.number_of_methods()
        
        if has_contracts:
            ci.methodcstats = ir.Struct(
                ir.Scoped_id(prefix, ci.epv.name+'__method_cstats', ''),
                [ir.Struct_item(ir.Typedef_type('int32_t'), 'tries'),
                 ir.Struct_item(ir.Typedef_type('int32_t'), 'successes'),
                 ir.Struct_item(ir.Typedef_type('int32_t'), 'failures'),
                 ir.Struct_item(ir.Typedef_type('int32_t'), 'nonvio_exceptions')],
                '')
            contract_cstats.append(ir.Struct_item(ir.Typedef_type('sidl_bool'), 'enabled'))
            contract_cstats.append(ir.Struct_item(
                    ci.methodcstats, 'method_cstats[%d]'%num_methods))

        ci.cstats = ir.Struct(
            ir.Scoped_id(prefix, ci.epv.name+'__cstats', ''),
            [ir.Struct_item(ir.Typedef_type('sidl_bool'), 'use_hooks')]+contract_cstats,
            'The controls and statistics structure')

        # @class@__object
        inherits = []
        def gen_inherits(baseclass):
            inherits.append(ir.Struct_item(
                babel.ir_object_type(baseclass[1], baseclass[2])
                [1], # not a pointer, it is an embedded struct
                'd_'+str.lower(babel.qual_id(baseclass))))

        with_sidl_baseclass = not cls.is_interface() and cls.qualified_name <> ['sidl', 'BaseClass']
        
        # pointers to the base class' EPV
        par = cls.get_parent()
        if par and par.is_class():
            gen_inherits(par.get_scoped_id())
            with_sidl_baseclass = False

        # pointers to the implemented interface's EPV
        if not cls.is_interface():
            for impl in cls.get_unique_interfaces():
                if impl <> (sidl.scoped_id, ('sidl',), 'BaseInterface', ''):
                    gen_inherits(impl)

        baseclass = []
        if with_sidl_baseclass:
            baseclass.append(
                ir.Struct_item(ir.Struct('sidl_BaseClass__object', [],''),
                               'd_sidl_baseclass'))

        if ior_template.generateContractEPVs(ci.co):
            cstats = [ir.Struct_item(unscope(ci.cstats), 'd_cstats')]

        ior_template.gen_hooks = self.config.gen_hooks
        ior_template.gen_contracts = self.config.gen_contracts
        epv  = [ir.Struct_item(ir.Pointer_type(unscope(ci.epv.get_type())), 'd_epv')]
        bepv = [ir.Struct_item(ir.Pointer_type(unscope(ci.epv.get_type())), 'd_bepv')] \
               if ior_template.generateBaseEPVAttr(ci.co) else []

        ci.obj = \
            ir.Struct(ir.Scoped_id(prefix, ci.epv.name+'__object', ''),
                      baseclass+
                      inherits+
                      epv+
                      bepv+
                      cstats+
                       [ir.Struct_item(ir.Pointer_type(ir.pt_void),
                                       'd_object' if cls.is_interface() else
                                       'd_data')],
                       'The class object structure')
        ci.external = \
            ir.Struct(ir.Scoped_id(prefix, ci.epv.name+'__external', ''),
                      ([ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Pointer_type(ci.obj),
                                                       'createObject', [
                                                           ir.Arg([], ir.inout, ir.void_ptr, 'ddata'),
                                                           ir.Arg([], sidl.out, babel.ir_exception_type(), '_ex')],
                                                       '')),
                                                       'createObject')] 
                      if not cls.is_abstract else []) +
                      ([ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Pointer_type(unscope(ci.epv.get_sepv_type())),
                                                       'getStaticEPV', [], '')),
                                                       'getStaticEPV')] 
                      if cls.has_static_methods else []) +
                      [ir.Struct_item(
                        ir.Pointer_type(
                            ir.Fn_decl([], ir.Pointer_type(ir.Struct('sidl_BaseClass__epv', [],'')),
                                       'getSuperEPV', [], '')),
                        'getSuperEPV'),
                       ir.Struct_item(ir.pt_int, 'd_ior_major_version'),
                       ir.Struct_item(ir.pt_int, 'd_ior_minor_version')
                       ],
                       'The static class object structure')
