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
# Contributors/Acknowledgements:
#
# Summer interns at LLNL:
# * 2010, 2011 Shams Imam <shams@rice.edu> 
#   contributed argument conversions, r-array handling, exception handling, 
#   distributed arrays, the patches to the Chapel compiler, ...
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

import config, ior, ior_template, ir, os, os.path, re, sidl, splicer, tempfile, types
from utils import *
from patmat import *
from cgen import *
from sidl_symbols import *

chpl_data_var_template = '_babel_data_{arg_name}'
chpl_dom_var_template = '_babel_dom_{arg_name}'
chpl_local_var_template = '_babel_local_{arg_name}'
chpl_param_ex_name = '_babel_param_ex'
extern_def_is_not_null = 'extern proc IS_NOT_NULL(in aRef): bool;'
extern_def_set_to_null = 'extern proc SET_TO_NULL(inout aRef);'
chpl_base_exception = 'BaseException'
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
    if ci.is_interface and args:
        _, attrs, type, id, arguments, doc = cdecl
        _, attrs0, mode0, type0, name0 = arguments[0]
        arguments = [ir.Arg([ir.pure], mode0, type0, name0)]+arguments[1:]
        cdecl = ir.Fn_decl(attrs, type, id, arguments, doc)
        
    return ir.Stmt(ir.Call(ir.Deref(ir.Get_struct_item(epv,
                ir.Deref(ir.Get_struct_item(ci.obj,
                                            ir.Deref('self'),
                                            ir.Struct_item(epv, 'd_epv'))),
                ir.Struct_item(ir.Pointer_type(cdecl), 'f_'+name))), args))

@accepts(str, str)
def write_to(filename, string):
    """
    Create/Overwrite a file named \c filename with the contents of \c
    string.
    The file is written atomically.
    """
    f = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.')
    f.write(string)
    #f.flush()
    #os.fsync(f)
    f.close()
    os.rename(f.name, filename)

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

class Chapel(object):
    
    class ClassInfo(object):
        """
        Holder object for the code generation scopes and other data
        during the traversal of the SIDL tree.
        """
        def __init__(self, name, symbol_table, is_interface, is_abstract,
                     has_static_fns,
                     stub_parent=None,
                     skel_parent=None):
            
            self.impl = ChapelFile()
            self.chpl_method_stub = ChapelFile(stub_parent, relative_indent=4)
            self.chpl_skel = ChapelFile(skel_parent, relative_indent=0)
            self.chpl_static_stub = ChapelFile(stub_parent)            
            self.chpl_static_skel = ChapelFile(skel_parent)            
            self.skel = CFile()
            self.epv = EPV(name, symbol_table, has_static_fns)
            self.ior = CFile()
            self.obj = None
            self.is_interface = is_interface
            self.is_abstract = is_abstract

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
            self.generate_server_pkg(self.sidl_ast, None, self.symbol_table)
            if self.create_makefile:
                generate_server_makefile(self.sidl_file, self.pkgs, self.classes)

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

        def generate_class_stub(name, methods, extends, doc_comment,
                                is_interface, implements=[]):
            """
            shared code for class/interface
            """
            # Qualified name (C Version)
            qname = '_'.join(symbol_table.prefix+[name])
            # Qualified name including Chapel modules
            mod_qname = '.'.join(symbol_table.prefix[1:]+[qname])
            mod_name = '.'.join(symbol_table.prefix[1:]+[name])

            self.has_static_methods = False

            # Consolidate all methods, defined and inherited
            all_names = set()
            all_methods = []

            is_abstract = member_chk(sidl.abstract, self.class_attrs)
            scan_methods(symbol_table, is_abstract, 
                         extends, implements, methods, 
                         all_names, all_methods, self, True)

            # Initialize all class-specific code generation data structures
            chpl_stub = ChapelFile()
            ci = self.ClassInfo(name, symbol_table, is_interface, 
                                is_abstract,
                                self.has_static_methods,
                                stub_parent=chpl_stub)
            chpl_stub.cstub.genh(ir.Import(qname+'_IOR'))
            chpl_stub.cstub.genh(ir.Import('sidlType'))
            chpl_stub.cstub.genh(ir.Import('chpl_sidl_array'))
            chpl_stub.cstub.genh(ir.Import('chpltypes'))
            if self.has_static_methods:
                chpl_stub.cstub.new_def(
                    externals((sidl.scoped_id, symbol_table.prefix, name, '')))

            self.gen_default_methods(symbol_table, extends, implements, ci)

            # recurse to generate method code
            #print qname, map(lambda x: x[2][1]+x[2][2], all_methods)
            gen1(all_methods, ci)

            # Stub (in Chapel)
            # Chapel supports C structs via the extern keyword,
            # but they must be typedef'ed in a header file that
            # must be passed to the chpl compiler.
            typedefs = self.class_typedefs(qname, symbol_table)
            write_to(qname+'_Stub.h', typedefs.dot_h(qname+'_Stub.h'))
            chpl_defs = chpl_stub
            chpl_stub = ChapelFile(chpl_defs)
            chpl_stub.new_def('use sidl;')
            extrns = ChapelScope(chpl_stub, relative_indent=0)

            def gen_extern_casts(baseclass):
                base = qual_id(baseclass)
                mod_base = '.'.join(baseclass[1]+[base])
                ex = 'out ex: sidl_BaseInterface__object'
                extrns.new_def('extern proc _cast_{0}(in ior: {1}__object, {2}): {3}__object;'
                               .format(base, mod_qname, ex, mod_base))
                extrns.new_def('extern proc {3}_cast_{1}(in ior: {0}__object): {2}__object;'
                               .format(mod_base, qname, mod_qname, base))

            parent_classes = []
            extern_hier_visited = []
            for _, ext in extends:
                sidl.visit_hierarchy(ext, gen_extern_casts, symbol_table, 
                                     extern_hier_visited)
                parent_classes += strip_common(symbol_table.prefix, ext[1])

            parent_interfaces = []
            for _, impl in implements:
                sidl.visit_hierarchy(impl, gen_extern_casts, symbol_table,  
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
            name = chpl_gen(name)
            
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
            if not ci.is_interface:
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
            if not ci.is_interface:
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

            def gen_cast(base):
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
            for _, ext in extends:
                sidl.visit_hierarchy(ext, gen_cast, symbol_table, casts_generated)
            for _, impl in implements:
                sidl.visit_hierarchy(impl, gen_cast, symbol_table, casts_generated)

            chpl_class.new_def(chpl_defs.get_defs())
            
            chpl_stub.new_def(chpl_defs.get_decls())
            chpl_stub.new_def('// All the static methods of class '+name)
            chpl_stub.new_def('module %s_static {'%name)
            chpl_stub.new_def(ci.chpl_static_stub.get_defs())
            chpl_stub.new_def(chpl_static_helper)
            chpl_stub.new_def('}')
            chpl_stub.new_def('')
            chpl_stub.new_def(gen_doc_comment(doc_comment, chpl_stub)+
                              'class %s %s %s {'%(name,inherits,interfaces))
            chpl_stub.new_def(chpl_class)
            chpl_stub.new_def(ci.chpl_method_stub)
            chpl_stub.new_def('}')
            
            # This is important for the chapel stub, but we generate
            # separate files for the cstubs
            self.pkg_chpl_stub.new_def(chpl_stub)


            # IOR
            self.generate_ior(ci, extends, implements, all_methods)
            write_to(qname+'_IOR.h', ci.ior.dot_h(qname+'_IOR.h'))

            # Stub (in C)
            cstub = chpl_stub.cstub
            cstub.genh_top(ir.Import(qname+'_IOR'))
            for code in cstub.optional:
                cstub.new_global_def(code)

            cstub.gen(ir.Import(qname+'_cStub'))

            # Stub Header
            write_to(qname+'_cStub.h', cstub.dot_h(qname+'_cStub.h'))
            # Stub C-file
            write_to(qname+'_cStub.c', cstub.dot_c())

            # Makefile
            self.classes.append(qname)

        if not symbol_table:
            raise Exception()

        with match(node):
            if (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                self.generate_client_method(symbol_table, node, data)

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                generate_class_stub(Name, Methods, Extends, DocComment, False, Implements)

            elif (sidl.struct, (Name), Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_stub.gen(ir.Type_decl(lower_ir(symbol_table, node)))
                self.pkg_enums_and_structs.append(node)

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_class_stub(Name, Methods, Extends, DocComment, is_interface=True)

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_stub.gen(ir.Type_decl(node))
                self.pkg_enums_and_structs.append(node)
                
            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel stub
                qname = '_'.join(symbol_table.prefix+[Name])
                pkg_symbol_table = symbol_table[sidl.Scoped_id([], Name, '')]
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

                pkg_h = CFile()
                pkg_h.genh(ir.Import('sidlType'))
                for es in self.pkg_enums_and_structs:
                    pkg_h.gen(ir.Type_decl(lower_ir(pkg_symbol_table, es, pkg_h)))
                write_to(qname+'.h', pkg_h.dot_h(qname+'.h'))


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

    def gen_default_methods(self, symbol_table, extends, implements, ci):
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
                inarg(sidl.pt_opaque, 'FIXMEinArgs'),
                inarg(sidl.pt_opaque, 'FIXMEoutArgs')])

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
        if not ci.is_interface:
            builtin(sidl.void, '_ctor', [])
            builtin(sidl.void, '_ctor2',
                    [(sidl.arg, [], sidl.in_, ir.void_ptr, 'private_data')])
            builtin(sidl.void, '_dtor', [])
            builtin(sidl.void, '_load', [])

        static_builtin(sidl.void, 'setHooks_static', 
                [inarg(sidl.pt_bool, 'enable')])
        static_builtin(sidl.void, 'set_contracts_static', [
                inarg(sidl.pt_bool, 'enable'),
                inarg(sidl.pt_string, 'enfFilename'),
                inarg(sidl.pt_bool, 'resetCounters')],
                )
        static_builtin(sidl.void, 'dump_stats_static', 
                [inarg(sidl.pt_string, 'filename'),
                 inarg(sidl.pt_string, 'prefix')])


        prefix = ci.epv.symbol_table.prefix
        # cstats
        cstats = []
        ci.cstats = ir.Struct(
            ir.Scoped_id(prefix, ci.epv.name+'__cstats', ''),
            [ir.Struct_item(ir.Typedef_type("sidl_bool"), "use_hooks")],
            'The controls and statistics structure')

        # @class@__object
        inherits = []
        def gen_inherits(baseclass):
            inherits.append(ir.Struct_item(
                ir_babel_object_type(baseclass[1], baseclass[2])
                [1], # not a pointer, it is an embedded struct
                'd_'+str.lower(qual_id(baseclass))))

        with_sidl_baseclass = not ci.is_interface and ci.epv.name <> 'BaseClass'
        
        # pointers to the base class' EPV
        for _, ext in extends:
            if ext[2] <> 'BaseInterface':
                gen_inherits(ext)
                with_sidl_baseclass = False

        # pointers to the implemented interface's EPV
        for _, impl in implements:
            gen_inherits(impl)

        baseclass = []
        if with_sidl_baseclass:
            baseclass.append(
                ir.Struct_item(ir.Struct('sidl_BaseClass__object', [],''),
                               "d_sidl_baseclass"))

        if not ci.is_interface and not ci.is_abstract:
            cstats = [ir.Struct_item(unscope(ci.cstats), "d_cstats")]

            
        ci.obj = \
            ir.Struct(ir.Scoped_id(prefix, ci.epv.name+'__object', ''),
                      baseclass+
                      inherits+
                      [ir.Struct_item(ir.Pointer_type(unscope(ci.epv.get_type())), "d_epv")]+
                       cstats+
                       [ir.Struct_item(ir.Pointer_type(ir.pt_void),
                                       'd_object' if ci.is_interface else
                                       'd_data')],
                       'The class object structure')
        ci.external = \
            ir.Struct(ir.Scoped_id(prefix, ci.epv.name+'__external', ''),
                      ([ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Pointer_type(ci.obj),
                                                       "createObject", [
                                                           ir.Arg([], ir.inout, ir.void_ptr, 'ddata'),
                                                           ir.Arg([], sidl.out, ir_babel_exception_type(), chpl_local_exception_var)],
                                                       "")),
                                                       "createObject")] 
                      if not ci.is_abstract else []) +
                      ([ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Pointer_type(unscope(ci.epv.get_sepv_type())),
                                                       "getStaticEPV", [], "")),
                                                       "getStaticEPV")] 
                      if ci.epv.has_static_fns else []) +
                      [ir.Struct_item(
                        ir.Pointer_type(
                            ir.Fn_decl([], ir.Pointer_type(ir.Struct('sidl_BaseClass__epv', [],'')),
                                       "getSuperEPV", [], "")),
                        "getSuperEPV"),
                       ir.Struct_item(ir.pt_int, "d_ior_major_version"),
                       ir.Struct_item(ir.pt_int, "d_ior_minor_version")
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
            'typedef struct sidl_BaseException__object _sidl_BaseException__object;',
            'typedef _sidl_BaseException__object* sidl_BaseException__object;',
            'typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;',
            'typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;',
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
        """

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
                    cname = name + '.self_' + typ[2]

                if mode <> sidl.in_:
                    cname = '_IOR_' + name
                    
                    pre_call.append(ir.Stmt(ir.Var_decl(ctype, cname)))
                    
                    # wrap the C type in a native Chapel object
                    chpl_class_name = typ[2]
                    mod_chpl_class_name = '.'.join(typ[1]+[chpl_class_name])
                    conv = ir.Call(qual_id(typ, '.') + '_static.wrap_' + chpl_class_name, 
                                   [cname, chpl_param_ex_name])
                    
                    if name == 'retval':
                        post_call.append(ir.Stmt(ir.Var_decl((ir.typedef_type, mod_chpl_class_name), name)))
                    post_call.append(ir.Stmt(ir.Assignment(name, conv)))
                    
                    if name == 'retval':
                        return_expr.append(name)
            
            elif is_struct_type(symbol_table, typ):
                ctype = ir.Pointer_type(symbol_table[typ])

            elif typ[0] == sidl.scoped_id:
                # Other Symbol
                ctype = symbol_table[typ]

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
                ctype = ir.Pointer_type(ir.Struct('sidl_%s__array'%t, [], ''))
                if mode <> sidl.out:
                    cname = name+'.self'

                if mode <> sidl.in_:
                    cname = '_IOR_' + name
                    # wrap the C type in a native Chapel object
                    pre_call.append(ir.Stmt(ir.Var_decl(ctype, cname)))
                    if mode == sidl.inout:
                        pre_call.append(ir.Stmt(ir.Assignment(cname, name+'.self')))

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
        
        chpl_args = []
        chpl_args.extend(lower_structs(symbol_table, Args))
        
        ci.epv.add_method((Method, Type, (MName,  Name, Extension), Attrs, ior_args,
                           Except, From, Requires, Ensures, DocComment))

        abstract = member_chk(sidl.abstract, Attrs)
        final = member_chk(sidl.final, Attrs)
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
        if ci.is_interface:
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
        _, (_,_,_,ctype,_) = convert_arg((ir.arg, [], ir.out, Type, 'retval'))

        cdecl_args = babel_stub_args(attrs, cdecl_args, symbol_table, ci.epv.name, docast)
        cdecl = ir.Fn_decl(attrs, ctype, Name + Extension, cdecl_args, DocComment)

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

        defn = (ir.fn_defn, [], lower_structs(symbol_table, Type), Name + Extension, chpl_args,
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
            # ci.chpl_static_stub.new_def('extern '+chpl_gen(extern_decl)+';')
            # chpl_stub.cstub.new_header_def('extern '+str(c_gen(impl_decl))+';')
            chpl_gen(defn, ci.chpl_static_stub)
        else:
            chpl_gen(defn, ci.chpl_method_stub)

    def generate_ior(self, ci, extends, implements, methods):
        """
        Generate the IOR header file in C.
        """
        prefix = '_'.join(ci.epv.symbol_table.prefix)
        cname = '_'.join([prefix, ci.epv.name])
        ci.ior.genh(ir.Import(prefix))
        ci.ior.genh(ir.Import('sidl'))
        ci.ior.genh(ir.Import('sidl_BaseInterface_IOR'))

        def gen_cast(scope):
            base = qual_id(scope)
            ci.ior.genh(ir.Import(base+'_IOR'))
            # Cast functions for the IOR
            ci.ior.genh('#define _cast_{0}(ior,ex) ((struct {0}__object*)((*ior->d_epv->f__cast)(ior,"{1}",ex)))'
                       .format(base, qual_id(scope, '.')))
            ci.ior.genh('#define {1}_cast_{0}(ior) ((struct {1}__object*)'
                       '((struct sidl_BaseInterface__object*)ior)->d_object)'
                       .format(cname, base))
        
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
            
            add_forward_defn(cname)
            refs = ['sidl_BaseException', 'sidl_BaseInterface']
            
            # lookup extends/impls clause
            for _, ext in extends:
                refs.append(qual_id(ext))

            for _, impl in implements:
                refs.append(qual_id(impl))
                    
            # lookup method args and return types
            for loop_method in methods:
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

        for _, ext in extends:
            gen_cast(ext)

        for _, impl in implements:
            gen_cast(impl)
                
        # FIXME Need to insert forward references to external structs (i.e. classes) used as return type/parameters        
                
        ci.ior.genh(ir.Import('stdint'))
        ci.ior.genh(ir.Import('chpl_sidl_array'))
        gen_forward_references()
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

        ci.ior.gen(ir.Fn_decl([], ir.pt_void, cname+'__init',
            babel_static_ior_args([], [ir.Arg([], ir.in_, ir.void_ptr, 'ddata')],
                           ci.epv.symbol_table, ci.epv.name),
            "INIT: initialize a new instance of the class object."))
        ci.ior.gen(ir.Fn_decl([], ir.pt_void, cname+'__fini',
            babel_static_ior_args([], [], ci.epv.symbol_table, ci.epv.name),
            "FINI: deallocate a class instance (destructor)."))
        ci.ior.new_def(ior_template.text.format(
            Class = cname, Class_low = str.lower(cname)))

    
    @matcher(globals(), debug=False)
    def generate_server_pkg(self, node, data, symbol_table):
        """
        SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER SERVER
        """
        def gen(node):         return self.generate_server_pkg(node, data, symbol_table)
        def gen1(node, data1): return self.generate_server_pkg(node, data1, symbol_table)

        def generate_class_stub(name, methods, extends, doc_comment,
                                is_interface, implements=[]):
            """
            shared code for class/interface
            """

            # Qualified name (C Version)
            qname = '_'.join(symbol_table.prefix+[name])  
            # Qualified name including Chapel modules
            mod_qname = '.'.join(symbol_table.prefix[1:]+[qname])  
            mod_name = '.'.join(symbol_table.prefix[1:]+[name])

            self.has_static_methods = False

            # Consolidate all methods, defined and inherited
            all_names = set()
            all_methods = []

            is_abstract = member_chk(sidl.abstract, self.class_attrs)
            scan_methods(symbol_table, is_abstract,
                         extends, implements, methods, 
                         all_names, all_methods, self, True)

            # Initialize all class-specific code generation data structures
            chpl_stub = ChapelFile()
            ci = self.ClassInfo(name, symbol_table, is_interface, 
                                is_abstract,
                                self.has_static_methods,
                                stub_parent=chpl_stub)
            ci.impl.gen(ir.Import('sidl'))
            ci.impl.gen(ir.Import('_'.join(symbol_table.prefix)))
            typedefs = self.class_typedefs(qname, symbol_table)
            ci.chpl_skel.cstub._header.extend(typedefs._header)
            ci.chpl_skel.cstub._defs.extend(typedefs._defs)
            chpl_stub.cstub.genh(ir.Import(qname+'_IOR'))
            chpl_stub.cstub.genh(ir.Import('sidlType'))
            chpl_stub.cstub.genh(ir.Import('chpl_sidl_array'))
            chpl_stub.cstub.genh(ir.Import('chpltypes'))
            if self.has_static_methods:
                chpl_stub.cstub.new_def(
                    externals((sidl.scoped_id, symbol_table.prefix, name, '')))

            self.gen_default_methods(symbol_table, extends, implements, ci)

            #print qname, map(lambda x: x[2][1]+x[2][2], all_methods)
            for method in all_methods:
                (Method, Type, Name, Attrs, Args, 
                 Except, From, Requires, Ensures, DocComment) = method
                ci.epv.add_method((method, Type, Name, Attrs, 
                                   drop_rarray_ext_args(Args),
                                   Except, From, Requires, Ensures, DocComment))

            # not very efficient...
            def builtin(t, name, args):
                methods.append(
                    sidl.Method(t, sidl.Method_name(name, ''), [],
                                args, [], [], [], [], 
                                'Implicit built-in method: '+name))
            builtin(sidl.void, '_ctor', [])
            builtin(sidl.void, '_ctor2',
                    [(sidl.arg, [], sidl.in_, ir.void_ptr, 'private_data')])
            builtin(sidl.void, '_dtor', [])

            # recurse to generate method code
            gen1(methods, ci) #all_methods

            self.generate_ior(ci, extends, implements, all_methods)

            # IOR
            write_to(qname+'_IOR.h', ci.ior.dot_h(qname+'_IOR.h'))
            write_to(qname+'_IOR.c', ci.ior.dot_c())

            # The server-side stub is used for, e.g., the
            # babelized Array-init functions

            # Stub (in C)
            cstub = chpl_stub.cstub
            cstub.gen(ir.Import(qname+'_cStub'))
            # Stub Header
            write_to(qname+'_cStub.h', cstub.dot_h(qname+'_cStub.h'))
            # Stub C-file
            write_to(qname+'_cStub.c', cstub.dot_c())

            # Skeleton (in Chapel)
            skel = ci.chpl_skel
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
            cskel.gen(ir.Import('stdint'))                
            cskel.gen(ir.Import(qname+'_Skel'))
            cskel.gen(ir.Import(qname+'_IOR'))
            cskel.gen(ir.Fn_defn([], ir.pt_void, qname+'__call_load', [],
                                   [ir.Comment("FIXME: [ir.Stmt(ir.Call('_load', []))")], ''))

            # set_epv ... Setup the EPV
            epv_t = ci.epv.get_ir()
            pre_epv_t = ci.epv.get_pre_epv_ir()
            post_epv_t = ci.epv.get_post_epv_ir()
            cskel.gen(ir.Fn_decl([], ir.pt_void, 'ctor', [], ''))
            cskel.gen(ir.Fn_decl([], ir.pt_void, 'dtor', [], ''))

            epv_init = []
            for m in methods:
                name = m[2][1]
                def entry(epv_t, table, field, pointer):
                    epv_init.append(ir.Set_struct_item_stmt(epv_t, ir.Deref(table), field, pointer))

                entry(epv_t, 'epv', 'f_'+name, name+'_impl')
                builtins = set(['_ctor', '_ctor2', '_dtor'])
                if name not in builtins:
                    entry(pre_epv_t, 'pre_epv', 'f_%s_pre'%name, 'NULL')
                    entry(post_epv_t, 'post_epv', 'f_%s_post'%name, 'NULL')
            
            cskel.gen(ir.Fn_defn(
                [], ir.pt_void, qname+'__set_epv',
                [ir.Arg([], ir.out, epv_t, 'epv'),
                 ir.Arg([], ir.out, pre_epv_t, 'pre_epv'),
                 ir.Arg([], ir.out, post_epv_t, 'post_epv')],
                epv_init, ''))

            write_to('_chplmain.c', chplmain_extras)

            # Skel Header
            write_to(qname+'_Skel.h', cskel.dot_h(qname+'_Skel.h'))
            # Skel C-file
            write_to(qname+'_Skel.c', cskel.dot_c())

            # Impl
            #write_to(qname+'_Impl.chpl', str(ci.impl))
            pkg_name = '_'.join(symbol_table.prefix)
            impl = pkg_name+'_Impl.chpl'
            if os.path.isfile(impl):
                # FIXME: possible race condition, should use a file
                # handle instead
                # Preserve code written by the user
                splicers = splicer.record(impl)
                write_to(impl, str(ci.impl))
                splicer.apply_all(impl, splicers)
            else:
                write_to(impl, str(ci.impl))

            # Makefile
            self.classes.append(qname)


        if not symbol_table:
            raise Exception()

        with match(node):
            if (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                self.generate_server_method(symbol_table, node, data)

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                generate_class_stub(Name, Methods, Extends, DocComment, False, Implements)

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_class_stub(Name, Methods, Extends, DocComment, is_interface=True)

            elif (sidl.struct, (Name), Items, DocComment):
                # record it for late
                self.pkg_enums_and_structs.append(node)

            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel skel
                qname = '_'.join(symbol_table.prefix+[Name])
                pkg_symbol_table = symbol_table[sidl.Scoped_id([], Name, '')]
                if self.in_package:
                    # nested modules are generated in-line
                    self.pkg_chpl_stub.new_def('module %s {'%Name)
                    self.generate_server_pkg(UserTypes, data, pkg_symbol_table)
                    self.pkg_chpl_stub.new_def('}')
                else:
                    # new file for the toplevel package
                    self.pkg_chpl_skel = ChapelFile()
                    self.pkg_chpl_skel.main_area.new_def('proc __defeat_dce(){\n')

                    self.pkg_enums_and_structs = []
                    self.in_package = True
                    self.generate_server_pkg(UserTypes, data, pkg_symbol_table)
                    self.pkg_chpl_skel.main_area.new_def('}\n')
                    write_to(qname+'_Skel.chpl', str(self.pkg_chpl_skel))

                pkg_chpl = ChapelFile()
                pkg_h = CFile()
                pkg_h.genh(ir.Import('sidlType'))
                for es in self.pkg_enums_and_structs:
                    pkg_h.gen(ir.Type_decl(lower_ir(pkg_symbol_table, es, pkg_h)))
                    symtab = pkg_symbol_table._parent
                    if symtab == None: symtab = SymbolTable()
                    pkg_chpl.gen(ir.Type_decl(lower_ir(symtab, es)))

                write_to(qname+'.h', pkg_h.dot_h(qname+'.h'))
                write_to(qname+'.chpl', str(pkg_chpl))

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
            cname = name
            ctype = typ
            return cname, (arg, attrs, mode, typ, name)


        # Chapel skeleton
        (Method, Type, (MName,  Name, Extension), Attrs, Args,
         Except, From, Requires, Ensures, DocComment) = method

        ior_args = drop_rarray_ext_args(Args)
        
#        ci.epv.add_method((Method, Type, (MName,  Name, Extension), Attrs, ior_args,
#                           Except, From, Requires, Ensures, DocComment))

        abstract = member_chk(sidl.abstract, Attrs)
        static = member_chk(sidl.static, Attrs)
        final = member_chk(sidl.static, Attrs)

        if abstract:
            # nothing to be done for an abstract function
            return

        pre_call = []
        post_call = []
        call_args, cdecl_args = unzip(map(convert_arg, ior_args))
        return_expr = []
        return_stmt = []
        #callee = ''.join(['.'.join(ci.epv.symbol_table.prefix+
        #                   [ci.epv.name]), '_Impl',
        #                  '_static' if static else '',
        #                  '.', Name, '_impl'])
        callee = Name+'_impl'

        if not static:
            call_args = ['self->d_data']+call_args

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

        ior_args = drop_rarray_ext_args(Args)

        defn = (ir.fn_defn, [], Type, Name,
                babel_epv_args(Attrs, Args, ci.epv.symbol_table, ci.epv.name),
                pre_call+call+post_call+return_stmt,
                DocComment)
        chpldecl = (ir.fn_decl, [], Type, callee,
                    [ir.Arg([], ir.in_, ir.void_ptr, '_this')]+lower_ir(ci.epv.symbol_table, Args),
                    DocComment)
        splicer = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name, Name])
        chpldefn = (ir.fn_defn, ['export %s'%callee], Type, callee,
                    [ir.Arg([], ir.in_, ir.void_ptr, '_this')]+Args,
                    [ir.Comment('DO-NOT-DELETE splicer.begin(%s)'%splicer),
                     ir.Comment('DO-NOT-DELETE splicer.end(%s)'%splicer)],
                    DocComment)

        c_gen(chpldecl, ci.chpl_skel.cstub)
        c_gen(defn, ci.chpl_skel.cstub)
        chpl_gen(chpldefn, ci.impl)

        ## create dummy call to bypass dead code elimination
        #def argvardecl((arg, attrs, mode, typ, name)):
        #    return ir.Var_decl(typ, name)
        #argdecls = map(argvardecl, Args)
        #def get_arg_name((arg, attrs, mode, typ, name)):
        #    return name
        #dcall = ir.Call(Name, [] if static else ['obj']+map(get_arg_name, Args)+[chpl_local_exception_var])
        #ci.chpl_skel.main_area.new_def('{\n')
        #ci.chpl_skel.main_area.new_def('var obj: %s__object;\n'%
        #                               '_'.join(ci.epv.symbol_table.prefix+[ci.epv.name]))
        #ci.chpl_skel.main_area.new_def('var ex: sidl_BaseInterface__object;\n')
        #chpl_gen(argdecls+[dcall], ci.chpl_skel.main_area)
        #ci.chpl_skel.main_area.new_def('}\n')



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
    def wrapped(*args):
        r = fn(*args)
        if r == None:
            print args
            print '---->', r
            raise Exception("lower_ir() output failed sanity check")
        return r

    return wrapped

@notnone
@matcher(globals(), debug=False)
def lower_ir(symbol_table, sidl_term, header=None):
    """
    FIXME!! can we merge this with convert_arg??
    lower SIDL types into IR

    The idea is that no Chapel-specific code is in this function. It
    should provide a generic translation from SIDL -> IR.
    """
    def low(sidl_term):
        return lower_ir(symbol_table, sidl_term, header)

#    print 'low(',sidl_term,')'
    with match(sidl_term):
        if (sidl.arg, Attrs, Mode, (sidl.scoped_id, _, _, _), Name):
            lowtype = low(sidl_term[3])
            if lowtype[0] == ir.struct:
                # struct arguments are passed as pointer, regardless of mode
                lowtype = ir.Pointer_type(lowtype)
            return ir.Arg(Attrs, Mode, lowtype, Name)

        elif (sidl.arg, Attrs, Mode, Typ, Name):
            return ir.Arg(Attrs, Mode, low(Typ), Name)

        elif (sidl.scoped_id, Prefix, Name, Ext):
            return low(symbol_table[sidl_term])
        
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
            return ir.Struct(qual_id(sidl_term[1]), low(Items), '')
        elif (sidl.struct, Name, Items, DocComment):
            #print 'Items=',Items, 'low(Items)=',low(Items)
            qname = '_'.join(symbol_table.prefix+[Name])
            return ir.Struct(qname, low(Items), '')

        elif (sidl.struct_item, Type, Name):
            return ir.Struct_item(low(Type), Name)

        # elif (sidl.rarray, Scalar_type, Dimension, Extents):
        #     # Direct-call version (r-array IOR)
        #     # return ir.Pointer_type(lower_type_ir(symbol_table, Scalar_type)) # FIXME
        #     # SIDL IOR version
        #     return ir.Typedef_type('sidl_%s__array'%Scalar_type[1])
        elif (sidl.rarray, Scalar_type, Dimension, Extents):
            # Rarray appearing inside of a struct
            return ir.Pointer_type(Scalar_type)

        # elif (sidl.array, Scalar_type, Dimension, Orientation):
        #     # array appearing inside of a struct
        #     if Scalar_type[0] == ir.scoped_id:
        #         ctype = 'BaseInterface'
        #     else:
        #         ctype = Scalar_type[1]
        #     return 'struct sidl_xs__array)'#%(gen(Scalar_type), ctype)

        # elif (sidl.rarray, Scalar_type, Dimension, Name, Extents):
        #     # r-array appearing inside of a struct
        #     return '/*FIXME*/ sidl_%s__array'%Scalar_type[1]
        #     #return gen(Scalar_type)

        elif (sidl.array, [], [], []):
            return ir.Pointer_type(ir.pt_void)

        elif (sidl.array, Scalar_type, Dimension, Orientation):
            #return ir.Typedef_type('sidl__array')
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

        elif (sidl.class_, ScopedId, _, _, _, _):
            return ir_babel_object_type(ScopedId[1], ScopedId[2])
        
        elif (sidl.interface, ScopedId, _, _, _):
            return ir_babel_object_type(ScopedId[1], ScopedId[2])
        
        elif (Terms):
            if (isinstance(Terms, list)):
                return map(low, Terms)
        else:
            raise Exception("lower_ir: Not implemented: " + str(sidl_term))

@matcher(globals(), debug=False)
def lower_structs(symbol_table, sidl_term):
    """
    FIXME paper hack ahead!!!
    """
    def low(sidl_term):
        return lower_structs(symbol_table, sidl_term)

    with match(sidl_term):   
        if (sidl.arg, Attrs, Mode, (sidl.scoped_id, _, _, _), Name):
            lowtype = low(sidl_term[3])
            if lowtype[0] == ir.struct:
                # struct arguments are passed as pointer, regardless of mode
                lowtype = ir.Pointer_type(lowtype)
            return (ir.arg, Attrs, Mode, lowtype, Name)

        elif (sidl.arg, Attrs, Mode, Typ, Name):
            return (ir.arg, Attrs, Mode, low(Typ), Name)

        elif (sidl.struct, (sidl.scoped_id, Prefix, Name, Ext), Items, DocComment):
            # a nested Struct
            return ir.Struct(qual_id(sidl_term[1]), low(Items), '')
        elif (sidl.struct, Name, Items, DocComment):
            #print 'Items=',Items, 'low(Items)=',low(Items)
            qname = '_'.join(symbol_table.prefix+[Name])
            return ir.Struct(qname, low(Items), '')

        elif (sidl.struct_item, Type, Name):
            return ir.Struct_item(lower_ir(symbol_table, Type), Name)

        elif (sidl.scoped_id, Prefix, Name, Ext):
            t = symbol_table[sidl_term]
            if t[0] == ir.struct:
                return low(t)
            return sidl_term

        elif (Terms):
            if (isinstance(Terms, list)):
                return map(low, Terms)
            
    return sidl_term



def get_type_name((fn_decl, Attrs, Type, Name, Args, DocComment)):
    return ir.Pointer_type((fn_decl, Attrs, Type, Name, Args, DocComment)), Name

class EPV(object):
    """
    Babel entry point vector for virtual method calls.

    Also contains the SEPV, which is used for all static functions, as
    well as the pre- and post-epv for the hooks implementation.

    """
    def __init__(self, name, symbol_table, has_static_fns):
        self.methods = []
        self.static_methods = []
        # hooks
        self.pre_methods = []
        self.post_methods = []
        self.static_pre_methods = []
        self.static_post_methods = []

        self.name = name
        self.symbol_table = symbol_table
        self.finalized = False
        self.has_static_fns = has_static_fns

    def add_method(self, method, with_hooks=False):
        """
        add another (SIDL) method to the vector
        """
        def to_fn_decl((_sidl_method, Type,
                        (Method_name, Name, Extension),
                        Attrs, Args, Except, From, Requires, Ensures, DocComment),
                       suffix=''):
            typ = lower_ir(self.symbol_table, Type)
            if typ[0] == ir.struct:
                typ = ir.Pointer_type(typ)
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
        return None

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

        if not self.has_static_fns:
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
        if not self.has_static_fns:
            return None

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
        if not self.has_static_fns:
            return None

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
        arg_self = extra_attrs
    else:
        arg_self = [
            ir.Arg(extra_attrs, sidl.in_, 
                ir_babel_object_type(symbol_table.prefix, class_name), 'self')]
    arg_ex = \
        [ir.Arg(extra_attrs, sidl.out, ir_babel_exception_type(), chpl_local_exception_var)]
    return arg_self+args+arg_ex


def is_obj_type(symbol_table, typ):
    return typ[0] == sidl.scoped_id and (
        symbol_table[typ][0] == sidl.class_ or
        symbol_table[typ][0] == sidl.interface)

def is_struct_type(symbol_table, typ):
    return typ[0] == sidl.scoped_id and symbol_table[typ][0] == sidl.struct


def ior_type(symbol_table, t):
    """
    if \c t is a scoped_id return the IOR type of t.
    else return \c t.
    """
    if (t[0] == sidl.scoped_id and symbol_table[t][0] in [sidl.class_, sidl.interface]):
    #    return ir_babel_object_type(*symbol_table.get_full_name(t[1]))
        return ir_babel_object_type(t[1], t[2])

    else: return t



def generate_client_makefile(sidl_file, classes):
    """
    FIXME: make this a file copy from $prefix/share
           make this work for more than one class
    """
    files = 'IORHDRS = '+' '.join([c+'_IOR.h' for c in classes])+'\n'
    files+= 'STUBHDRS = '+' '.join(['{c}_Stub.h {c}_cStub.h'.format(c=c)
                                    for c in classes])+'\n'
    files+= 'STUBSRCS = '+' '.join([c+'_cStub.c' for c in classes])+'\n'
# this is handled by the use statement in the implementation instead:
# {file}_Stub.chpl
    write_to('babel.make', files)
    generate_client_server_makefile(sidl_file)


def generate_server_makefile(sidl_file, pkgs, classes):
    """
    FIXME: make this a file copy from $prefix/share
           make this work for more than one class
    """
    write_to('babel.make', """
IMPLHDRS =
IMPLSRCS = {impls}
IORHDRS = {iorhdrs} #FIXME Array_IOR.h
IORSRCS = {iorsrcs}
SKELSRCS = {skelsrcs}
STUBHDRS = #FIXME {stubhdrs}
STUBSRCS = {stubsrcs}
""".format(impls=' '.join([p+'_Impl.chpl'       for p in pkgs]),
           iorhdrs=' '.join([c+'_IOR.h'    for c in classes]),
           iorsrcs=' '.join([c+'_IOR.c'    for c in classes]),
           skelsrcs=' '.join([c+'_Skel.c'  for c in classes]),
           stubsrcs=' '.join([c+'_cStub.c' for c in classes]),
           stubhdrs=' '.join([c+'_Stub.h'  for c in classes])))
    generate_client_server_makefile(sidl_file)

def generate_client_server_makefile(sidl_file):
    extraflags=''
    #extraflags='-ggdb -O0'
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
EXTRAFLAGS="""+extraflags+r"""
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
CHPL_MAKE_HOME="""+config.CHAPEL_ROOT+r"""
CHAPEL_MAKE_COMM="""+config.CHAPEL_COMM+r"""

CC=`babel-config --query-var=CC`
INCLUDES=`babel-config --includes` -I. -I$(CHPL_MAKE_HOME)/runtime/include -I$(SIDL_RUNTIME)/chpl
CFLAGS=`babel-config --flags-c` -std=c99
LIBS=`babel-config --libs-c-client`

CHAPEL_MAKE_MEM=default
CHAPEL_MAKE_COMPILER=gnu
CHAPEL_MAKE_TASKS=none
CHAPEL_MAKE_THREADS=pthreads

ifeq ($(CHAPEL_MAKE_COMM),gasnet)
CHAPEL_MAKE_SUBSTRATE_DIR=$(CHPL_MAKE_HOME)/lib/$(CHPL_HOST_PLATFORM)/$(CHAPEL_MAKE_COMPILER)/mem-default/comm-gasnet-nodbg/substrate-udp/seg-none
else
CHAPEL_MAKE_SUBSTRATE_DIR=$(CHPL_MAKE_HOME)/lib/$(CHPL_HOST_PLATFORM)/$(CHAPEL_MAKE_COMPILER)/mem-default/comm-none/substrate-none/seg-none
endif
####    include $(CHPL_MAKE_HOME)/runtime/etc/Makefile.include
CHPL=chpl --fast
# CHPL=chpl --print-commands --print-passes

include $(CHPL_MAKE_HOME)/make/Makefile.atomics

CHPL_FLAGS=-std=c99 -DCHPL_TASKS_MODEL_H=\"tasks-fifo.h\" -DCHPL_THREADS_MODEL_H=\"threads-pthreads.h\" -I$(CHPL_MAKE_HOME)/runtime/include/tasks/fifo -I$(CHPL_MAKE_HOME)/runtime/include/threads/pthreads -I$(CHPL_MAKE_HOME)/runtime/include/comm/none -I$(CHPL_MAKE_HOME)/runtime/include/comp-gnu -I$(CHPL_MAKE_HOME)/runtime/include/$(CHPL_HOST_PLATFORM) -I$(CHPL_MAKE_HOME)/runtime/include/atomics/$(CHPL_MAKE_ATOMICS) -I$(CHPL_MAKE_HOME)/runtime/include -I. -Wno-all 

CHPL_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm  -lpthread -lsidlstub_chpl

CHPL_GASNET_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm -lpthread -L$(CHPL_MAKE_HOME)/third-party/gasnet/install/$(CHPL_HOST_PLATFORM)-$(CHAPEL_MAKE_COMPILER)/seg-everything/nodbg/lib -lgasnet-udp-par -lamudp -lpthread -lgcc -lm

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
  DCE=--no-dead-code-elimination # include everything in libimpl.la
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
          -std=c99 -I$(CHPL_MAKE_HOME)/runtime/include/$(CHPL_HOST_PLATFORM) \
	  -I$(CHPL_MAKE_HOME)/runtime/include -I. \
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

STUBOBJS=$(patsubst .chpl, .lo, $(STUBSRCS:.c=.lo))
IOROBJS=$(IORSRCS:.c=.lo)
SKELOBJS=$(SKELSRCS:.c=.lo)
IMPLOBJS=$(IMPLSRCS:.chpl=.lo)

PUREBABELGEN=$(IORHDRS) $(IORSRCS) $(STUBSRCS) $(STUBHDRS) $(SKELSRCS)
BABELGEN=$(IMPLHDRS) $(IMPLSRCS)

$(IMPLOBJS) : $(STUBHDRS) $(IORHDRS) $(IMPLHDRS)

lib$(LIBNAME).la : $(STUBOBJS) $(IOROBJS) $(IMPLOBJS) $(SKELOBJS)
	babel-libtool --mode=link --tag=CC $(CC) -o lib$(LIBNAME).la \
	  -static \
          -release $(VERSION) \
	  -no-undefined $(MODFLAG) \
	  $(CFLAGS) $(EXTRAFLAGS) $^ $(LIBS) \
          $(CHPL_LDFLAGS) -lchpl \
	  $(EXTRALIBS)
 #-rpath $(LIBDIR) 

$(PUREBABELGEN) $(BABELGEN) : babel-stamp
# cf. http://www.gnu.org/software/automake/manual/automake.html#Multiple-Outputs
# Recover from the removal of $@
	@if test -f $@; then :; else \
	  trap 'rm -rf babel.lock babel-stamp' 1 2 13 15; \
true "mkdir is a portable test-and-set"; \
	  if mkdir babel.lock 2>/dev/null; then \
true "This code is being executed by the first process."; \
	    rm -f babel-stamp; \
	    $(MAKE) $(AM_MAKEFLAGS) babel-stamp; \
	    result=$$?; rm -rf babel.lock; exit $$result; \
	  else \
true "This code is being executed by the follower processes."; \
true "Wait until the first process is done."; \
	    while test -d babel.lock; do sleep 1; done; \
true "Succeed if and only if the first process succeeded." ; \
	    test -f babel-stamp; \
	  fi; \
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
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; fi ; \
	echo '  <library uri="'`pwd`/lib$(LIBNAME).la'" scope="'"$$scope"'" resolution="lazy" >' >> $@
	grep __set_epv $^ /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >>$@
	echo "  </library>" >>$@
	echo "</scl>" >>$@
endif

.SUFFIXES: .lo .chpl

.c.lo:
	babel-libtool --mode=compile --tag=CC $(CC) $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) -c -o $@ $<

ifeq ($(IMPLSRCS),)
.chpl.lo:
	$(CHPL) --savec $<.dir $< $(IORHDRS) $(STUBHDRS) $(CHPL_HEADERS) $(DCE) --make true  # gen C-code only
	babel-libtool --mode=compile --tag=CC $(CC) \
            -I./$<.dir $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) \
            $(CHPL_FLAGS) -c -o $@ $<.dir/_main.c
else
.chpl.lo:
	$(CHPL) --library --savec $<.dir $< $(IORHDRS) $(STUBHDRS) $(CHPL_HEADERS) $(DCE) --make true  # gen C-code
	#headerize $<.dir/_config.c $<.dir/Chapel*.c $<.dir/Default*.c $<.dir/DSIUtil.c $<.dir/chpl*.c $<.dir/List.c $<.dir/Math.c $<.dir/Search.c $<.dir/Sort.c $<.dir/Types.c
	#perl -pi -e 's/((chpl__autoDestroyGlobals)|(chpl_user_main)|(chpl__init)|(chpl_main))/$*_\1/g' $<.dir/$*.c
	perl -pi -e 's|^  if .$*|  chpl_bool $*_chpl__init_$*_p = false;\n  if ($*|' $<.dir/$*.c
	echo '#include "../_chplmain.c"' >>$<.dir/_main.c
	babel-libtool --mode=compile --tag=CC $(CC) \
            -I./$<.dir $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) \
            $(CHPL_FLAGS) -c -o $@ $<.dir/_main.c
endif


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
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; fi ; \
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

