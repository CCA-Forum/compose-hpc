#!/usr/bin/env python
# -*- python -*-
## @package chapel.backend
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
# Copyright (c) 2011, Lawrence Livermore National Security, LLC.
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

import ior, ior_template, ir, os.path, sidl, sidlobjects, splicer
from utils import write_to, unzip
from patmat import *
from codegen import (CFile)
from sidl_symbols import visit_hierarchy
import conversions as conv
import makefile
from cgen import (ChapelFile, ChapelScope, chpl_gen, c_gen, 
                  incoming, outgoing, gen_doc_comment, strip, deref)

chpl_data_var_template = '_babel_data_{arg_name}'
chpl_dom_var_template = '_babel_dom_{arg_name}'
chpl_local_var_template = '_babel_local_{arg_name}'
chpl_param_ex_name = '_babel_param_ex'
extern_def_is_not_null = 'extern proc IS_NOT_NULL(in aRef): bool;'
extern_def_set_to_null = 'extern proc SET_TO_NULL(inout aRef);'
chpl_base_interface = 'BaseInterface'
chpl_local_exception_var = '_ex'
chplmain_extras = r"""
int handleNonstandardArg(int* argc, char* argv[], int argNum, 
                         int32_t lineno, chpl_string filename) {
  char* message = chpl_glom_strings(3, "Unexpected flag:  \"", argv[argNum], 
                                    "\"");
  chpl_error(message, lineno, filename);
  return 0;
}

void printAdditionalHelp(void) {
}

char* chpl_executionCommand;

static void recordExecutionCommand(int argc, char *argv[]) {
  int i, length = 0;
  for (i = 0; i < argc; i++) {
    length += strlen(argv[i]) + 1;
  }
  chpl_executionCommand = (char*)chpl_mem_allocMany(length+1, sizeof(char), CHPL_RT_EXECUTION_COMMAND, 0, 0);
  sprintf(chpl_executionCommand, "%s", argv[0]);
  for (i = 1; i < argc; i++) {
    strcat(chpl_executionCommand, " ");
    strcat(chpl_executionCommand, argv[i]);
  }
}
"""


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


class GlueCodeGenerator(object):
    """
    This class provides the methods to transform SIDL to IR.
    """
    
    class ClassInfo(object):
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
            self.epv = EPV(class_object)
            self.ior = CFile('_'.join(class_object.qualified_name)+'_IOR')
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
        self.pkgs = []

    def generate_client(self):
        """
        Generate client code. Operates in two passes:
        \li create symbol table
        \li do the work
        """
        try:
            self.generate_client_pkg(self.sidl_ast, None, self.symbol_table)
            if self.verbose:
                import sys
                sys.stdout.write("%s\r"%(' '*80))
            if self.create_makefile:
                makefile.generate_client(self.sidl_file, self.classes)
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
            self.generate_server_pkg(self.sidl_ast, None, self.symbol_table)
            if self.create_makefile:
                makefile.generate_server(self.sidl_file, self.pkgs, self.classes)

        except:
            # Invoke the post-mortem debugger
            import pdb, sys
            print sys.exc_info()
            pdb.post_mortem()


    @matcher(globals(), debug=False)
    def generate_client_pkg(self, node, data, symbol_table):
        """
        CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT CLIENT
        """
        def gen(node):         return self.generate_client_pkg(node, data, symbol_table)
        def gen1(node, data1): return self.generate_client_pkg(node, data1, symbol_table)

        def generate_ext_stub(cls):
            """
            shared code for class/interface
            """
            # Qualified name (C Version)
            qname = '_'.join(symbol_table.prefix+[cls.name])
            # Qualified name including Chapel modules
            mod_qname = '.'.join(symbol_table.prefix[1:]+[qname])
            mod_name = '.'.join(symbol_table.prefix[1:]+[cls.name])

            if self.verbose:
                import sys
                sys.stdout.write('\r'+' '*80)
                sys.stdout.write('\rgenerating glue code for %s'%mod_name)
                sys.stdout.flush()

            cls.scan_methods()
                
            # Initialize all class-specific code generation data structures
            chpl_stub = ChapelFile()
            chpl_defs = ChapelScope(chpl_stub)
            ci = self.ClassInfo(cls, stub_parent=chpl_stub)

            chpl_stub.cstub.genh(ir.Import(qname+'_IOR'))
            chpl_stub.cstub.genh(ir.Import('sidlType'))
            chpl_stub.cstub.genh(ir.Import('chpl_sidl_array'))
            chpl_stub.cstub.genh(ir.Import('chpltypes'))
            if cls.has_static_methods:
                chpl_stub.cstub.new_def(
                    externals((sidl.scoped_id, symbol_table.prefix, cls.name, '')))

            has_contracts = ior_template.generateContractChecks(cls)

            self.gen_default_methods(cls, has_contracts, ci)

            # recurse to generate method code
            #print qname, map(lambda x: x[2][1]+x[2][2], all_methods)
            gen1(cls.all_methods, ci)

            # Stub (in Chapel)
            # Chapel supports C structs via the extern keyword,
            # but they must be typedef'ed in a header file that
            # must be passed to the chpl compiler.
            typedefs = self.class_typedefs(qname, symbol_table)
            write_to(qname+'_Stub.h', typedefs.dot_h(qname+'_Stub.h'))
            chpl_stub.new_def('use sidl;')
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
                parent_classes += strip_common(symbol_table.prefix, ext[1])

            parent_interfaces = []
            for _, impl in cls.implements:
                visit_hierarchy(impl, gen_extern_casts, symbol_table,  
                                extern_hier_visited)
                parent_interfaces += strip_common(symbol_table.prefix, impl[1])

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
                vcall('addRef', ['this.' + self_field_name, 'ex'], ci),
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
                    'create_' + name,
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
            wrapped_obj_arg = ir.Arg([], ir.in_, ir_babel_object_type([], qname), 'obj')
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
            destructor_body.append(vcall('deleteRef', ['this.' + self_field_name, 'ex'], ci))
            if not cls.is_interface():
                # Interfaces have no destructor
                destructor_body.append(vcall('_dtor', ['this.' + self_field_name, 'ex'], ci))
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
                     'Create a down-casted version of the IOR pointer for\n'
                     'use with the alternate constructor'), chpl_class)

            gen_self_cast()
            casts_generated = [symbol_table.prefix+[name]]
            for _, ext in cls.extends:
                visit_hierarchy(ext, gen_cast, symbol_table, casts_generated)
            for _, impl in cls.implements:
                visit_hierarchy(impl, gen_cast, symbol_table, casts_generated)

            chpl_class.new_def(chpl_defs.get_defs())
            
            chpl_stub.new_def(chpl_defs.get_decls())
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
            
            # Because of chapels implicit (filename-based)modules it
            # is important for the chapel stub to be one file, but we
            # generate separate files for the cstubs
            self.pkg_chpl_stub.new_def(chpl_stub)

            # Stub (in C), the order is somewhat sensitive
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
            self.generate_ior(ci, with_ior_c=False)
            ci.ior.write()


            # Makefile
            self.classes.append(qname)

        if not symbol_table:
            raise Exception()

        with match(node):
            if (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                self.generate_client_method(symbol_table, node, data)

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                generate_ext_stub(sidlobjects.Class(symbol_table, node, self.class_attrs))

            elif (sidl.struct, (Name), Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_stub.gen(ir.Type_decl(lower_ir(symbol_table, node, struct_suffix='')))
                self.pkg_enums_and_structs.append(struct_ior_names(node))

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_ext_stub(sidlobjects.Interface(symbol_table, node, self.class_attrs))

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_stub.gen(ir.Type_decl(node))
                self.pkg_enums_and_structs.append(node)
                
            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel stub
                qname = '_'.join(symbol_table.prefix+[Name])
                _, pkg_symbol_table = symbol_table[sidl.Scoped_id([], Name, '')]
                if self.in_package:
                    # nested modules are generated in-line
                    self.pkg_chpl_stub.new_def('module %s {'%Name)
                    self.generate_client_pkg(UserTypes, data, pkg_symbol_table)
                    self.pkg_chpl_stub.new_def('}')
                else:
                    # new file for the toplevel package
                    self.pkg_chpl_stub = ChapelFile(relative_indent=0)
                    self.pkg_enums_and_structs = []
                    self.in_package = True
                    self.generate_client_pkg(UserTypes, data, pkg_symbol_table)
                    write_to(qname+'.chpl', str(self.pkg_chpl_stub))
     
                    # Makefile
                    self.pkgs.append(qname)

                pkg_h = CFile(qname)
                pkg_h.genh(ir.Import('sidlType'))
                pkg_h.genh(ir.Import('chpltypes'))
                for es in self.pkg_enums_and_structs:
                    es_ior = lower_ir(pkg_symbol_table, es, header=pkg_h)
                    pkg_h.gen(ir.Type_decl(es_ior))
                    # generate also the chapel version of the struct, if different
                    if es[0] == sidl.struct:
                        es_chpl = conv.ir_type_to_chpl(es_ior)
                        if es_chpl <> es_ior: 
                            pkg_h.gen(ir.Type_decl(es_chpl))

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
                inarg(babel_object_type(['sidl', 'rmi'], 'Call'), 'inArgs'),
                inarg(babel_object_type(['sidl', 'rmi'], 'Return'), 'outArgs')])

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
                ir.Scoped_id(prefix, ci.epv.name+'__methods_cstats', ''),
                [ir.Struct_item(ir.Typedef_type('int32_t'), 'tries'),
                 ir.Struct_item(ir.Typedef_type('int32_t'), 'successes'),
                 ir.Struct_item(ir.Typedef_type('int32_t'), 'failures'),
                 ir.Struct_item(ir.Typedef_type('int32_t'), 'nonvio_exceptions')],
                '')
            contract_cstats.append(ir.Struct_item(ir.Typedef_type('sidl_bool'), 'enabled'))
            contract_cstats.append(ir.Struct_item(
                    ci.methodcstats,  '_'.join(cls.qualified_name+['_method_cstats'])
                    +'[%d]'%num_methods))

        ci.cstats = ir.Struct(
            ir.Scoped_id(prefix, ci.epv.name+'__cstats', ''),
            [ir.Struct_item(ir.Typedef_type('sidl_bool'), 'use_hooks')]+contract_cstats,
            'The controls and statistics structure')

        # @class@__object
        inherits = []
        def gen_inherits(baseclass):
            inherits.append(ir.Struct_item(
                ir_babel_object_type(baseclass[1], baseclass[2])
                [1], # not a pointer, it is an embedded struct
                'd_'+str.lower(qual_id(baseclass))))

        with_sidl_baseclass = not cls.is_interface() and cls.qualified_name <> ['sidl', 'BaseClass']
        
        # pointers to the base class' EPV
        for _, ext in cls.extends:
            if ext[2] <> 'BaseInterface':
                gen_inherits(ext)
                with_sidl_baseclass = False

        # pointers to the implemented interface's EPV
        for _, impl in cls.implements:
            gen_inherits(impl)

        baseclass = []
        if with_sidl_baseclass:
            baseclass.append(
                ir.Struct_item(ir.Struct('sidl_BaseClass__object', [],''),
                               'd_sidl_baseclass'))

        if not cls.is_interface() and not cls.is_abstract:
            cstats = [ir.Struct_item(unscope(ci.cstats), 'd_cstats')]

            
        ci.obj = \
            ir.Struct(ir.Scoped_id(prefix, ci.epv.name+'__object', ''),
                      baseclass+
                      inherits+
                      [ir.Struct_item(ir.Pointer_type(unscope(ci.epv.get_type())), 'd_epv')]+
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
                                                           ir.Arg([], sidl.out, ir_babel_exception_type(), chpl_local_exception_var)],
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

    def class_typedefs(self, qname, symbol_table):
        typedefs = CFile();
        typedefs._header = [
            '// Package header (enums, etc...)',
            '#include <stdint.h>',
            '#include <%s.h>' % '_'.join(symbol_table.prefix),
            '#include <%s_IOR.h>'%qname,
            'typedef struct %s__object _%s__object;'%(qname, qname),
            'typedef _%s__object* %s__object;'%(qname, qname),
            '#ifndef SIDL_BASE_INTERFACE_OBJECT',
            '#define SIDL_BASE_INTERFACE_OBJECT',
            'typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;',
            'typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;',
            '#define IS_NULL(aPtr) ((aPtr) == 0)',
            '#define IS_NOT_NULL(aPtr) ((aPtr) != 0)',
            '#define SET_TO_NULL(aPtr) (*aPtr) = 0',
            '#endif',
            '%s__object %s__createObject(%s__object copy, sidl_BaseInterface__object* ex);'
            %(qname, qname, qname),
            ]
        return typedefs


    @matcher(globals(), debug=False)
    def generate_client_method(self, symbol_table, method, ci):
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
            if is_obj_type(symbol_table, typ):
                iortype = ior_type(symbol_table, typ)
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
            
            elif is_struct_type(symbol_table, typ):
                iortype = lower_ir(*symbol_table[typ])
                if (mode == sidl.in_):
                    iortype = ir.Pointer_type(iortype)

            elif typ[0] == sidl.scoped_id:
                # Other Symbol
                iortype = symbol_table[typ][1]

            elif typ == sidl.void:
                iortype = ir.pt_void

            elif typ == sidl.opaque:
                iortype = ir.Pointer_type(ir.pt_void)

            elif typ == (sidl.array, [], [], []): # Generic array
                iortype = ir.Pointer_type(ir.Struct('sidl__array', [], ''))

            elif typ[0] == sidl.array: # Scalar_type, Dimension, Orientation
                if typ[1][0] == ir.scoped_id:
                    t = 'BaseInterface'
                else:
                    t = typ[1][1]
                iortype = ir.Pointer_type(ir.Struct('sidl_%s__array'%t, [], ''))
                if mode <> sidl.out:
                    iorname = name+'.self'

                if mode <> sidl.in_:
                    iorname = '_IOR_' + name
                    # wrap the C type in a native Chapel object
                    pre_call.append(ir.Stmt(ir.Var_decl(iortype, iorname)))
                    if mode == sidl.inout:
                        pre_call.append(ir.Stmt(ir.Assignment(iorname, name+'.self')))

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
            var {a}rank = _babel_dom_{arg}.rank;

            var {a}lus = computeLowerUpperAndStride(_babel_local_{arg});
            var {a}lower = {a}lus(0);
            var {a}upper = {a}lus(1);
            var {a}stride = {a}lus(2);
            
            var _babel_wrapped_local_{arg}: {iortype} = {iortype}_borrow(
                {stype}_ptr(_babel_local_{arg}(_babel_local_{arg}.domain.low)),
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
            if is_obj_type(symbol_table, typ):
                return (arg, attrs, sidl.in_, typ, name)
            else:
                return (arg, attrs, mode, typ, name)

        # Chapel stub
        (Method, Type, (MName,  Name, Extension), Attrs, Args,
         Except, From, Requires, Ensures, DocComment) = method

        ior_args = drop_rarray_ext_args(Args)
        
        chpl_args = []
        chpl_args.extend(lower_ir(symbol_table, Args, struct_suffix='', lower_scoped_ids=False))
        
        ci.epv.add_method((Method, Type, (MName,  Name, Extension), Attrs, ior_args,
                           Except, From, Requires, Ensures, DocComment))

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
        pre_call.append(ir.Stmt(ir.Var_decl(ir_babel_exception_type(), chpl_local_exception_var)))
        pre_call.append(ir.Stmt(ir.Call("SET_TO_NULL", [chpl_local_exception_var])))
        
        post_call.append(ir.Stmt(ir.If(
            ir.Call("IS_NOT_NULL", [chpl_local_exception_var]),
            [
                ir.Stmt(ir.Assignment(chpl_param_ex_name,
                                   ir.Call("new " + chpl_base_interface, [chpl_local_exception_var])))
            ]
        )))
        
        call_args, cdecl_args = unzip(map(convert_arg, ior_args))
        
        # return value type conversion -- treat it as an out argument
        _, (_,_,_,iortype,_) = convert_arg((ir.arg, [], ir.out, Type, '_retval'))

        cdecl_args = babel_stub_args(attrs, cdecl_args, symbol_table, ci.epv.name, docast)
        cdecl = ir.Fn_decl(attrs, iortype, Name + Extension, cdecl_args, DocComment)

        if static:
            call_self = []
        else:
            call_self = ['this.self_' + ci.epv.name]
                

        call_args = call_self + call_args + [chpl_local_exception_var]
        # Add the exception to the chapel method signature
        chpl_args.append(ir.Arg([], ir.out, (ir.typedef_type, chpl_base_interface), chpl_param_ex_name))
        

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
                ir.Struct_item(ir.Pointer_type(cdecl), 'f_' + Name + Extension))
            
        else:
            # dynamic virtual method call
            epv_type = ci.epv.get_type()
            obj_type = ci.obj
            callee = ir.Deref(ir.Get_struct_item(
                epv_type,
                ir.Deref(ir.Get_struct_item(obj_type,
                                            ir.Deref('self'),
                                            ir.Struct_item(epv_type, 'd_epv'))),
                ir.Struct_item(ir.Pointer_type(cdecl), 'f_' + Name + Extension)))


        if Type == sidl.void:
            Type = ir.pt_void
            call = [ir.Stmt(ir.Call(callee, call_args))]
        else:
            if return_expr or post_call:
                rvar = '_IOR__retval'
                if not return_expr:                                
                    if iortype[0] == ir.struct:
                        iortype = lower_ir(*symbol_table[Type], struct_suffix='')

                    pre_call.append(ir.Stmt(ir.Var_decl(iortype, rvar)))
                    rx = rvar
                else:
                    rx = return_expr[0]
                    
                call = [ir.Stmt(ir.Assignment(rvar, ir.Call(callee, call_args)))]
                return_stmt = [ir.Stmt(ir.Return(rx))]
            else:
                call = [ir.Stmt(ir.Return(ir.Call(callee, call_args)))]


        defn = (ir.fn_defn, [], 
                lower_ir(symbol_table, Type, struct_suffix='', lower_scoped_ids=False),
                Name + Extension, chpl_args,
                pre_call+call+post_call+return_stmt,
                DocComment)

        if static:
            chpl_gen(defn, ci.chpl_static_stub)
        else:
            chpl_gen(defn, ci.chpl_method_stub)



    def generate_ior(self, ci, with_ior_c):
        """
        Generate the IOR header file in C and the IOR C file.
        """
        prefix = '_'.join(ci.epv.symbol_table.prefix)
        iorname = '_'.join([prefix, ci.epv.name])
        ci.ior.genh(ir.Import(prefix))
        ci.ior.genh(ir.Import('sidl'))
        ci.ior.genh(ir.Import('sidl_BaseInterface_IOR'))

        def gen_cast(scope):
            """
            this is Chapel-specific... should we move it somewhere else?
            """
            base = qual_id(scope)
            ci.ior.genh(ir.Import(base+'_IOR'))
            # Cast functions for the IOR
            ci.ior.genh('#define _cast_{0}(ior,ex) ((struct {0}__object*)((*ior->d_epv->f__cast)(ior,"{1}",ex)))'
                       .format(base, qual_id(scope, '.')))
            ci.ior.genh('#define {1}_cast_{0}(ior) ((struct {1}__object*)'
                       '((struct sidl_BaseInterface__object*)ior)->d_object)'
                       .format(iorname, base))
        
        def gen_forward_references():
            
            def get_ctype(typ):
                """
                Extract name and generate argument conversions
                """
                ctype = ''
                if not typ:
                    pass
                elif typ[0] == sidl.scoped_id:
                    # Symbol
                    ctype = qual_id(typ)
                elif typ[0] == sidl.array: # Scalar_type, Dimension, Orientation
                    ctype = get_ctype(typ[1])    
                elif typ[0] == sidl.rarray: # Scalar_type, Dimension, ExtentsExpr
                    ctype = get_ctype(typ[1]) 
    
                return ctype
            
            def add_forward_defn(name):
                ci.ior.genh('struct ' + name + '__array;')
                ci.ior.genh('struct ' + name + '__object;')
            
            add_forward_defn(iorname)
            refs = ['sidl_BaseException', 'sidl_BaseInterface', 
                    'sidl_rmi_Call', 'sidl_rmi_Return']
            
            # lookup extends/impls clause
            for _, ext in ci.co.extends:
                refs.append(qual_id(ext))

            for _, impl in ci.co.implements:
                refs.append(qual_id(impl))
                    
            # lookup method args and return types
            for loop_method in ci.co.all_methods:
                (_, _, _, _, loop_args, _, _, _, _, _) = loop_method
                for loop_arg in loop_args:
                    (_, _, _, loop_typ, _) = loop_arg
                    loop_ctype = get_ctype(loop_typ)
                    if loop_ctype:
                        refs.append(loop_ctype)
            
            # lookup static function args and return types
            
            # sort the refs and the add the forward references to the header
            refs = list(set(refs))
            refs.sort()
            for loop_ref in refs:
                add_forward_defn(loop_ref)

        for _, ext in ci.co.extends:
            gen_cast(ext)

        for _, impl in ci.co.implements:
            gen_cast(impl)
                
        # FIXME Need to insert forward references to external structs (i.e. classes) used as return type/parameters        
                
        ci.ior.genh(ir.Import('stdint'))
        ci.ior.genh(ir.Import('chpl_sidl_array'))
        ci.ior.genh(ir.Import('chpltypes'))
        gen_forward_references()
        if ci.methodcstats: 
            ci.ior.gen(ir.Type_decl(ci.methodcstats))
        ci.ior._header.extend(ior_template.contract_decls(ci.co, iorname))
        ci.ior.gen(ir.Type_decl(ci.cstats))
        ci.ior.gen(ir.Type_decl(ci.obj))
        ci.ior.gen(ir.Type_decl(ci.external))
        ci.ior.gen(ir.Type_decl(ci.epv.get_ir()))
        ci.ior.gen(ir.Type_decl(ci.epv.get_pre_epv_ir()))
        ci.ior.gen(ir.Type_decl(ci.epv.get_post_epv_ir()))

        sepv = ci.epv.get_sepv_ir() 
        if sepv:
            ci.ior.gen(ir.Type_decl(sepv))
            ci.ior.gen(ir.Type_decl(ci.epv.get_pre_sepv_ir()))
            ci.ior.gen(ir.Type_decl(ci.epv.get_post_sepv_ir()))

        ci.ior.gen(ir.Fn_decl([], ir.pt_void, iorname+'__init',
            babel_static_ior_args([], [ir.Arg([], ir.in_, ir.void_ptr, 'ddata')],
                           ci.epv.symbol_table, ci.epv.name),
            "INIT: initialize a new instance of the class object."))
        ci.ior.gen(ir.Fn_decl([], ir.pt_void, iorname+'__fini',
            babel_static_ior_args([], [], ci.epv.symbol_table, ci.epv.name),
            "FINI: deallocate a class instance (destructor)."))

        if with_ior_c:# or True:
            ci.ior.new_def(ior_template.gen_IOR_c(iorname, ci.co))

    
    @matcher(globals(), debug=False)
    def generate_server_pkg(self, node, data, symbol_table):
        """
        SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER
        """
        def gen(node):         return self.generate_server_pkg(node, data, symbol_table)
        def gen1(node, data1): return self.generate_server_pkg(node, data1, symbol_table)

        def generate_ext_stub(cls):
            """
            shared code for class/interface
            """

            # Qualified name (C Version)
            qname = '_'.join(cls.qualified_name)  

            # Consolidate all methods, defined and inherited
            cls.scan_methods()

            # Initialize all class-specific code generation data structures
            chpl_stub = ChapelFile()
            ci = self.ClassInfo(cls, stub_parent=chpl_stub)

            ci.impl = self.pkg_impl
            ci.impl.new_def(gen_doc_comment(cls.doc_comment, chpl_stub)+
                            'class %s_Impl {'%qname)

            typedefs = self.class_typedefs(qname, cls.symbol_table)
            chpl_stub.cstub._header.extend(typedefs._header)
            chpl_stub.cstub._defs.extend(typedefs._defs)
            chpl_stub.cstub.genh(ir.Import(qname+'_IOR'))
            chpl_stub.cstub.genh(ir.Import('sidlType'))
            chpl_stub.cstub.genh(ir.Import('chpl_sidl_array'))
            chpl_stub.cstub.genh(ir.Import('chpltypes'))
            if cls.has_static_methods:
                chpl_stub.cstub.new_def(externals(cls.get_scoped_id()))

            has_contracts = ior_template.generateContractChecks(cls)
            self.gen_default_methods(cls, has_contracts, ci)

            #print qname, map(lambda x: x[2][1]+x[2][2], all_methods)
            for method in cls.all_methods:
                (Method, Type, Name, Attrs, Args, 
                 Except, From, Requires, Ensures, DocComment) = method
                ci.epv.add_method((method, Type, Name, Attrs, 
                                   drop_rarray_ext_args(Args),
                                   Except, From, Requires, Ensures, DocComment))

            # recurse to generate method implementation skeletons
            gen1(builtins+cls.get_methods(), ci)

            ci.impl.new_def('}')

            # IOR
            self.generate_ior(ci, with_ior_c=True)
            ci.ior.write()

            # The server-side stub is used for thinks like the
            # babelized Array-init functions

            # Stub (in C)
            cstub = chpl_stub.cstub
            cstub._name = qname+'_cStub'
            cstub.gen(ir.Import(cstub._name))
            cstub.write()

            # Skeleton (in Chapel)
            self.pkg_chpl_skel.gen(ir.Import('.'.join(symbol_table.prefix)))

            self.pkg_chpl_skel.new_def('use sidl;')
            objname = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name]) + '_Impl'

            self.pkg_chpl_skel.new_def('extern record %s__object { var d_data: %s; };'
                                       %(qname,objname))
            self.pkg_chpl_skel.new_def('extern proc %s__createObject('%qname+
                                 'd_data: int, '+
                                 'out ex: sidl_BaseInterface__object)'+
                                 ': %s__object;'%qname)
            self.pkg_chpl_skel.new_def(ci.chpl_skel)


            # Skeleton (in C)
            cskel = ci.chpl_skel.cstub
            cskel._name = qname+'_Skel'
            cskel.gen(ir.Import('stdint'))                
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
                name =  m[2][1] + m[2][2]
                attrs = sidl.method_method_attrs(m)
                static = member_chk(sidl.static, attrs)
                def entry(stmts, epv_t, table, field, pointer):
                    stmts.append(ir.Set_struct_item_stmt(epv_t, ir.Deref(table), field, pointer))

                if static: entry(sepv_init, sepv_t, 'sepv', 'f_'+name, name+'_skel')
                else:      entry(epv_init,   epv_t,  'epv', 'f_'+name, name+'_skel')

                builtin_names = ['_ctor', '_ctor2', '_dtor']
                with_hooks = member_chk(ir.hooks, attrs)
                if name not in builtin_names and with_hooks:
                    if static: entry(sepv_init, pre_sepv_t,  'pre_sepv',  'f_%s_pre'%name,  'NULL')
                    else:      entry(epv_init,  pre_epv_t,   'pre_epv',   'f_%s_pre'%name,  'NULL')
                    if static: entry(sepv_init, post_sepv_t, 'post_sepv', 'f_%s_post'%name, 'NULL')
                    else:      entry(epv_init,  post_epv_t,  'post_epv',  'f_%s_post'%name, 'NULL')
            
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

            write_to('_chplmain.c', chplmain_extras)

            # C Skel
            for code in cskel.optional:
                cskel.new_global_def(code)
            cskel.write()

            # Makefile
            self.classes.append(qname)


        if not symbol_table:
            raise Exception()

        with match(node):
            if (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                self.generate_server_method(symbol_table, node, data)

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                generate_ext_stub(sidlobjects.Class(symbol_table, node, self.class_attrs))

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_ext_stub(sidlobjects.Interface(symbol_table, node, self.class_attrs))

            elif (sidl.struct, (Name), Items, DocComment):
                # record it for later
                self.pkg_enums_and_structs.append(struct_ior_names(node))

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_skel.gen(ir.Type_decl(node))
                self.pkg_enums_and_structs.append(node)

            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel skel
                qname = '_'.join(symbol_table.prefix+[Name])
                _, pkg_symbol_table = symbol_table[sidl.Scoped_id([], Name, '')]
                if self.in_package:
                    # nested modules are generated in-line
                    self.pkg_chpl_skel.new_def('module %s {'%Name)
                    self.generate_server_pkg(UserTypes, data, pkg_symbol_table)
                    self.pkg_chpl_skel.new_def('}')
                else:
                    # new file for the toplevel package
                    self.pkg_chpl_skel = ChapelFile(qname+'_Skel')
                    self.pkg_chpl_skel.main_area.new_def('proc __defeat_dce(){\n')

                    # new file for the user implementation
                    self.pkg_impl = ChapelFile(qname+'_Impl')
                    self.pkg_impl.gen(ir.Import('sidl'))
                    self.pkg_impl.gen(ir.Import(qname))

                    self.pkg_enums_and_structs = []
                    self.in_package = True
                    self.generate_server_pkg(UserTypes, data, pkg_symbol_table)
                    self.pkg_chpl_skel.main_area.new_def('}\n')
                    self.pkg_chpl_skel.write()

                    # write the _Impl file
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

                # write the package-wide definitions (enums, structs)
                pkg_chpl = ChapelFile(qname)
                pkg_h = CFile(qname)
                pkg_h.genh(ir.Import('sidlType'))
                for es in self.pkg_enums_and_structs:
                    es_ior = lower_ir(pkg_symbol_table, es, header=pkg_h)
                    es_chpl = es_ior
                    if es[0] == sidl.struct:
                        es_ior = conv.ir_type_to_chpl(es_ior)
                        es_chpl = conv.ir_type_to_chpl(es)

                    pkg_h.gen(ir.Type_decl(es_ior))
                    pkg_chpl.gen(ir.Type_decl(es_chpl))

                pkg_h.write()
                pkg_chpl.write()

                # Makefile
                self.pkgs.append(qname)

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
        ior_args = lower_ir(symbol_table, Args, lower_scoped_ids=False)
        ctype = lower_ir(symbol_table, Type, lower_scoped_ids=False)
        return_stmt = []
        skel = ci.chpl_skel
        opt = skel.cstub.optional
        callee = '%s_impl'%'_'.join(ci.co.qualified_name+[Name])
     
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

        def pointerize_struct((arg, attr, mode, typ, name)):
          # FIXME: this is borked.. instead we should remove this
          # _and_ the code in codegenerater that strips the
          # pointer_type again
          if typ[0] == ir.struct:
                return (arg, attr, mode, (ir.pointer_type, typ), name)
          else: return (arg, attr, mode, typ, name)

        chpl_args = map(pointerize_struct, map(conv.ir_arg_to_chpl, ior_args))

     
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
                if mode <> sidl.in_ or is_struct:
                    name = ir.Pointer_expr(name)
     
            if name == 'self' and member_chk(ir.pure, attrs):
                # part of the hack for self dereferencing
                upcast = ('({0}*)(((struct sidl_BaseInterface__object*)self)->d_object)'
                          .format(c_gen(c_t[1])))
                call_args.append(upcast)
            else:
                if is_retval: is_retval = False
                else:         call_args.append(name)


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

        skeldefn = (ir.fn_defn, [], ctype, Name+'_skel',
                    babel_epv_args(Attrs, Args, ci.epv.symbol_table, ci.epv.name),
                    decls+pre_call+call+post_call+return_stmt,
                    DocComment)

        def lower_array_args((arg, attr, mode, typ, name)):
            if typ[0] == sidl.array:
                return arg, attr, mode, ir.pt_void, name
            else: return arg, attr, mode, typ, name

        impldecl = (ir.fn_decl, [], chpltype, callee,
                    this_arg+map(lower_array_args, chpl_args),
                    DocComment)
        splicer = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name, Name])
        impldefn = (ir.fn_defn, ['export '+callee], 
                    Type, Name,
                    this_arg+Args,
                    [ir.Comment('DO-NOT-DELETE splicer.begin(%s)'%splicer),
                     ir.Comment('DO-NOT-DELETE splicer.end(%s)'%splicer)],
                    DocComment)

        c_gen(skeldefn, ci.chpl_skel.cstub)
        c_gen(impldecl, ci.chpl_skel.cstub)
        chpl_gen(impldefn, ci.impl)




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

'''.format(a=qual_id(scopedid), b=qual_id(scopedid, '_'))

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
        # was: ir.Typedef_type('int64_t')
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
            if not lower_scoped_ids: return sidl_term #qual_id(ScopedId, '.')
            else: return ir_babel_object_type(ScopedId[1], ScopedId[2])
        
        elif (sidl.interface, ScopedId, _, _, _, _):
            if not lower_scoped_ids: sidl_term #qual_id(ScopedId, '.')
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


def babel_static_ior_args(attrs, args, symbol_table, class_name):
    """
    \return a SIDL -> Ir lowered version of 
    [self]+args+(sidl_BaseInterface__object*)[*ex]
    """
    arg_self = [ir.Arg([], ir.in_, 
                       ir_babel_object_type(symbol_table.prefix, class_name),
                       'self')]
    arg_ex = [ir.Arg([], sidl.out, ir_babel_baseinterface_type(), chpl_local_exception_var)]
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
        [ir.Arg([], ir.out, ir_babel_baseinterface_type(), chpl_local_exception_var)]
    return arg_self+lower_ir(symbol_table, args)+arg_ex

def babel_stub_args(attrs, args, symbol_table, class_name, extra_attrs=[]):
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
        [ir.Arg(extra_attrs, sidl.out, ir_babel_exception_type(), chpl_local_exception_var)]
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


