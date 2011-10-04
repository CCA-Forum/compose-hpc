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

import config, ior_template, ir, os, re, sidl, tempfile, types
from lists import *
from patmat import *
from cgen import *

chpl_data_var_template = '_babel_data_{arg_name}'
chpl_dom_var_template = '_babel_dom_{arg_name}'
chpl_local_var_template = '_babel_local_{arg_name}'
chpl_param_ex_name = '_babel_param_ex'
extern_def_is_not_null = '_extern proc IS_NOT_NULL(in aRef): bool;'
extern_def_set_to_null = '_extern proc SET_TO_NULL(inout aRef);'
chpl_base_exception = 'BaseException'
chpl_local_exception_var = '_ex'

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
    return babel_object_type(['sidl'], 'BaseException')

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
    return ir_babel_object_type(['sidl'], 'BaseException')

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
            self.generate_server1(self.sidl_ast, None, self.symbol_table)
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

            def scan_class(extends, implements, methods):
                """
                Recursively resolve the inheritance hierarchy
                """

                def full_method_name(method):
                    """
                    Return the long name of a method (sans class/packages)
                    for sorting purposes.
                    """
                    return method[2][1]+method[2][2]

                def add_method(m):
                    if member_chk(sidl.static, sidl.method_method_attrs(m)):
                        self.has_static_methods = True

                    if not full_method_name(m) in all_names:
                        all_names.add(full_method_name(m))
                        all_methods.append(m)

                def remove_method(m):
                    """
                    If we encounter a overloaded method with an extension,
                    we need to insert that new full name into the EPV, but
                    we also need to remove the original definition of that
                    function from the EPV.
                    """
                    (_, _, (_, name, _), _, args, _, _, _, _, _) = m

                    i = 0
                    for i in range(len(all_methods)):
                        m = all_methods[i]
                        (_, _, (_, name1, _), _, args1, _, _, _, _, _) = m
                        if (name1, args1) == (name, args):
                            del all_methods[i]
                            all_names.remove(full_method_name(m))
                            break

                def scan_protocols(implements):
                    for impl in implements:
                        for m in symbol_table[impl[1]][4]:
                            add_method(m)

                for _, ext in extends:
                    base = symbol_table[ext]
                    if base[0] == sidl.class_:
                        scan_class(sidl.class_extends(base), 
                                   sidl.class_implements(base), 
                                   sidl.class_methods(base))
                    elif base[0] == sidl.interface:
                        scan_class(sidl.interface_extends(base), 
                                   [], 
                                   sidl.interface_methods(base))
                    else: raise("?")
                    #scan_protocols(base[3])
                    #for m in base[5]:
                    #    add_method(m)

                scan_protocols(implements)

                for m in methods:
                    if m[6]: # from clause
                        remove_method(m)
                    add_method(m)

            scan_class(extends, implements, methods)

            # Initialize all class-specific code generation data structures
            chpl_stub = ChapelFile()
            ci = self.ClassInfo(name, symbol_table, is_interface, 
                                member_chk(sidl.abstract, self.class_attrs),
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
            # Chapel supports C structs via the _extern keyword,
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
                ex = 'inout ex: sidl_BaseException__object'
                extrns.new_def('_extern proc _cast_{0}(in ior: {1}__object, {2}): {3}__object;'
                               .format(base, mod_qname, ex, mod_base))
                extrns.new_def('_extern proc {3}_cast_{1}(in ior: {0}__object): {2}__object;'
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
            chpl_stub.new_def('_extern record %s__object {'%qname)
            chpl_stub.new_def('};')

            chpl_stub.new_def(extrns)
            chpl_stub.new_def('_extern proc %s__createObject('%qname+
                                 'd_data: int, '+
                                 'inout ex: sidl_BaseException__object)'+
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
                '  var ex: sidl_BaseException__object;',
                '  SET_TO_NULL(ex);'
            ]
            common_tail = [
                vcall('addRef', ['this.' + self_field_name, 'ex'], ci),
                '  if (IS_NOT_NULL(ex)) {',
                '     {arg_name} = new {base_ex}(ex);'.format(arg_name=chpl_param_ex_name, base_ex=chpl_base_exception) ,
                '  }'
            ]

            # The create() method to create a new IOR instance
            create_body = []
            create_body.extend(common_head)
            create_body.append('  this.' + self_field_name + ' = %s__createObject(0, ex);' % qname)
            create_body.extend(common_tail)
            wrapped_ex_arg = ir.Arg([], ir.inout, (ir.typedef_type, chpl_base_exception), chpl_param_ex_name)
            if not ci.is_interface:
                # Interfaces instances cannot be created!
                chpl_gen(
                    (ir.fn_defn, [], ir.pt_void,
                     'init_' + name,
                     [wrapped_ex_arg],
                     create_body, 'Psuedo-Constructor to initialize the IOR object'), chpl_class)
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
            destructor_body.append('var ex: sidl_BaseException__object;')
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
                     ['var ex: sidl_BaseException__object;',
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

            elif (sidl.interface, (Name), Extends, Invariants, Methods, DocComment):
                # Interfaces also have an IOR to be generated
                expect(data, None)
                generate_class_stub(Name, Methods, Extends, DocComment, is_interface=True)

            elif (sidl.enum, Name, Items, DocComment):
                # Generate Chapel stub
                self.pkg_chpl_stub.gen(ir.Type_decl(node))
                self.pkg_enums.append(node)
                
            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel stub
                qname = '_'.join(symbol_table.prefix+[Name])                
                if self.in_package:
                    # nested modules are generated in-line
                    self.pkg_chpl_stub.new_def('module %s {'%Name)
                    self.generate_client_pkg(UserTypes, data, symbol_table[
                            sidl.Scoped_id([], Name, '')])
                    self.pkg_chpl_stub.new_def('}')
                else:
                    # new file for the toplevel package
                    self.pkg_chpl_stub = ChapelFile(relative_indent=0)
                    self.pkg_enums = []
                    self.in_package = True
                    self.generate_client_pkg(UserTypes, data, symbol_table[
                            sidl.Scoped_id([], Name, '')])
                    write_to(qname+'.chpl', str(self.pkg_chpl_stub))
     
                    # Makefile
                    self.pkgs.append(qname)

                pkg_h = CFile()
                for enum in self.pkg_enums:
                    pkg_h.gen(ir.Type_decl(enum))
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
                'd_inherit_'+baseclass[2]))

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
                      [ir.Struct_item(ir.Pointer_type(ir.Fn_decl([],
                                                       ir.Pointer_type(ci.obj),
                                                       "createObject", [
                                                           ir.Arg([], ir.inout, ir.void_ptr, 'ddata'),
                                                           ir.Arg([], sidl.inout, ir_babel_exception_type(), chpl_local_exception_var)],
                                                       "")),
                                                       "createObject")]+
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
            '#define IS_NOT_NULL(aPtr) ((aPtr) != 0)',
            '#define SET_TO_NULL(aPtr) (*aPtr) = 0',
            '#endif',
            '%s__object %s__createObject(%s__object copy, sidl_BaseException__object* ex);'
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
            
            elif typ[0] == sidl.scoped_id:
                # Symbol
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
                ctype = ir.Typedef_type('sidl_%s__array'%t)
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
                
                # sanity check on input arra:yensure domain is rectangular
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
        chpl_args.extend(Args)
        
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
                                   ir.Call("new " + chpl_base_exception, [chpl_local_exception_var])))
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
        chpl_args.append(ir.Arg([], ir.inout, (ir.typedef_type, chpl_base_exception), chpl_param_ex_name))
        

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

        defn = (ir.fn_defn, [], Type, Name + Extension, chpl_args,
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

        sepv = ci.epv.get_sepv_ir() 
        if sepv:
            ci.ior.gen(ir.Type_decl(sepv))

        ci.ior.gen(ir.Fn_decl([], ir.pt_void, cname+'__init',
            babel_epv_args([], [ir.Arg([], ir.inout, ir.void_ptr, 'data')],
                           ci.epv.symbol_table, ci.epv.name),
            "INIT: initialize a new instance of the class object."))
        ci.ior.gen(ir.Fn_decl([], ir.pt_void, cname+'__fini',
            babel_epv_args([], [], ci.epv.symbol_table, ci.epv.name),
            "FINI: deallocate a class instance (destructor)."))
        ci.ior.new_def(ior_template.text.format(
            Class = cname, Class_low = str.lower(cname)))

    
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
                self.generate_server_method(symbol_table, node, data)

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                qname = '_'.join(symbol_table.prefix+[Name])                
                ci = self.ClassInfo(Name, symbol_table, False, None, self.pkg_chpl_skel)
                ci.chpl_skel.cstub.genh(ir.Import(qname+'_IOR'))
                self.gen_default_methods(symbol_table, Extends, Implementes, ci)
                gen1(Methods, ci)
                self.generate_ior(ci, Extends, Implements, Methods)

                # IOR
                write_to(qname+'_IOR.h', ci.ior.dot_h(qname+'_IOR.h'))
                write_to(qname+'_IOR.c', ci.ior.dot_c())

                # The server-side stub is used for, e.g., the
                # babelized Array-init functions

                # Stub (in C)
                cstub = ci.chpl_stub.cstub
                cstub.gen(ir.Import(qname+'_cStub'))
                # Stub Header
                write_to(qname+'_cStub.h', cstub.dot_h(qname+'_cStub.h'))
                # Stub C-file
                write_to(qname+'_cStub.c', cstub.dot_c())

                # Skeleton (in Chapel)
                skel = ci.chpl_skel
                self.pkg_chpl_skel.gen(ir.Import('.'.join(symbol_table.prefix)))

                typedefs = self.class_typedefs(qname, symbol_table)
                write_to(qname+'_Skel.h', typedefs.dot_h(qname+'_Skel.h'))

                self.pkg_chpl_skel.new_def('use sidl;')
                objname = '.'.join(ci.epv.symbol_table.prefix+[ci.epv.name]) + '_Impl'

                self.pkg_chpl_skel.new_def('_extern record %s__object { var d_data: %s; };'
                                           %(qname,objname))
                self.pkg_chpl_skel.new_def('_extern proc %s__createObject('%qname+
                                     'd_data: int, '+
                                     'inout ex: sidl_BaseException__object)'+
                                     ': %s__object;'%qname)
                self.pkg_chpl_skel.new_def(ci.chpl_skel)


                # Skeleton (in C)
                cskel = ci.chpl_skel.cstub
                cskel.gen(ir.Import('stdint'))                
                cskel.gen(ir.Import(qname+'_Skel'))
                cskel.gen(ir.Import(qname+'_IOR'))
                cskel.gen(ir.Fn_defn([], ir.pt_void, qname+'__call_load', [],
                                       [ir.Stmt(ir.Call('_load', []))], ''))
                epv_t = ci.epv.get_ir()
                cskel.gen(ir.Fn_decl([], ir.pt_void, 'ctor', [], ''))
                cskel.gen(ir.Fn_decl([], ir.pt_void, 'dtor', [], ''))
                cskel.gen(ir.Fn_defn(
                    [], ir.pt_void, qname+'__set_epv',
                    [ir.Arg([], ir.out, epv_t, 'epv'),
                     ir.Arg([], ir.out, epv_t, 'pre_epv'),
                     ir.Arg([], ir.out, epv_t, 'post_epv')],
                    [ir.Set_struct_item_stmt(epv_t, ir.Deref('epv'), 'f__ctor', 'ctor'),
                     ir.Set_struct_item_stmt(epv_t, ir.Deref('epv'), 'f__ctor2', '0'),
                     ir.Set_struct_item_stmt(epv_t, ir.Deref('epv'), 'f__dtor', 'dtor')], ''))
                
                # Skel Header
                write_to(qname+'_Skel.h', cskel.dot_h(qname+'_Skel.h'))
                # Skel C-file
                write_to(qname+'_Skel.c', cskel.dot_c())

                # Impl
                print "FIXME: update the impl file between the splicer blocks"
                #write_to(qname+'_Impl.chpl', str(ci.impl))

                # Makefile
                self.classes.append(qname)

            elif (sidl.package, Name, Version, UserTypes, DocComment):
                # Generate the chapel skel
                self.pkg_chpl_skel = ChapelFile()
                self.pkg_chpl_skel.main_area.new_def('proc __defeat_dce(){\n')

                self.pkg_enums = []
                self.generate_server1(UserTypes, data, symbol_table[sidl.Scoped_id([], Name, '')])
                self.pkg_chpl_skel.main_area.new_def('}\n')
                qname = '_'.join(symbol_table.prefix+[Name])                
                write_to(qname+'_Skel.chpl', str(self.pkg_chpl_skel))

                pkg_h = CFile()
                for enum in self.pkg_enums:
                    pkg_h.gen(ir.Type_decl(enum))
                write_to(qname+'.h', pkg_h.dot_h(qname+'.h'))

                # Makefile
                self.pkgs.append(qname)

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

        ci.epv.add_method((Method, Type, (MName,  Name, Extension), Attrs, ior_args,
                           Except, From, Requires, Ensures, DocComment))

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

        defn = (ir.fn_defn, [], Type, Name,
                babel_epv_args(Attrs, Args, ci.epv.symbol_table, ci.epv.name),
                pre_call+call+post_call+return_stmt,
                DocComment)
        chpldecl = (ir.fn_decl, [], Type, callee,
                [ir.Arg([], ir.in_, ir.void_ptr, 'this')]+Args,
                DocComment)
        c_gen(chpldecl, ci.chpl_skel.cstub)
        c_gen(defn, ci.chpl_skel.cstub)

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
        #ci.chpl_skel.main_area.new_def('var ex: sidl_BaseException__object;\n')
        #chpl_gen(argdecls+[dcall], ci.chpl_skel.main_area)
        #ci.chpl_skel.main_area.new_def('}\n')



def externals(scopedid):
    return '''
#include "sidlOps.h"

// Hold pointer to IOR functions.
static const struct {a}__external *_externals = NULL;

extern const struct {a}__external* {a}__externals();

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

        elif (sidl.void):               return ir.pt_void
        elif (sidl.primitive_type, _):  return low_t(sidl_term)
        elif (sidl.scoped_id, _, _, _): return low_t(sidl_term)
        elif (sidl.array, _, _, _):     return low_t(sidl_term)
        elif (sidl.rarray, _, _, _):    return low_t(sidl_term)
        
        elif (Terms):
            if (isinstance(Terms, list)):
                return map(low, Terms)
        else:
            raise Exception("lower_ir:: Not implemented: " + str(sidl_term))

@matcher(globals(), debug=False)
def lower_type_ir(symbol_table, sidl_type):
    """
    FIXME!! can we merge this with convert_arg??
    lower SIDL types into IR
    """
    with match(sidl_type):
        if (sidl.scoped_id, Prefix, Name, Ext):
            return lower_type_ir(symbol_table, symbol_table[sidl_type])
        
        elif (sidl.void):                        return ir.pt_void
        elif (ir.void_ptr):                      return ir.void_ptr
        elif (sidl.primitive_type, sidl.opaque): return ir.Pointer_type(ir.pt_void)
        elif (sidl.primitive_type, sidl.string): return ir.const_str
        elif (sidl.primitive_type, sidl.bool):   return ir.Typedef_type('sidl_bool')
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

        elif (sidl.class_, ScopedId, _, _, _, _):
            return ir_babel_object_type(ScopedId[1], ScopedId[2])
        
        elif (sidl.interface, ScopedId, _, _, _):
            return ir_babel_object_type(ScopedId[1], ScopedId[2])
        
        else:
            raise Exception("Not implemented: " + str(sidl_type))

def get_type_name((fn_decl, Attrs, Type, Name, Args, DocComment)):
    return ir.Pointer_type((fn_decl, Attrs, Type, Name, Args, DocComment)), Name

class EPV(object):
    """
    Babel entry point vector for virtual method calls.
    Also contains the SEPV which is used for all static functions.
    """
    def __init__(self, name, symbol_table, has_static_fns):
        self.methods = []
        self.static_methods = []
        self.name = name
        self.symbol_table = symbol_table
        self.finalized = False
        self.has_static_fns = has_static_fns

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

            # discard the abstract/final attributes. Ir doesn't know them.
            attrs = set(Attrs)
            attrs.discard(sidl.abstract)
            attrs.discard(sidl.final)
            attrs = list(attrs)
            args = babel_epv_args(attrs, Args, self.symbol_table, self.name)
            return ir.Fn_decl(attrs, typ, name, args, DocComment)

        if self.finalized:
            import pdb; pdb.set_trace()

        if member_chk(sidl.static, method[3]):
            self.static_methods.append(to_fn_decl(method))
        else:
            self.methods.append(to_fn_decl(method))
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
        [ir.Arg([], sidl.inout, ir_babel_exception_type(), chpl_local_exception_var)]
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
        [ir.Arg(extra_attrs, sidl.inout, ir_babel_exception_type(), chpl_local_exception_var)]
    return arg_self+args+arg_ex


def is_obj_type(symbol_table, typ):
    return typ[0] == sidl.scoped_id and (
        symbol_table[typ][0] == sidl.class_ or
        symbol_table[typ][0] == sidl.interface)


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
""".format(impls=' '.join([p+'.chpl'       for p in pkgs]),
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

CC=`babel-config --query-var=CC`
INCLUDES=`babel-config --includes` -I. -I$(CHAPEL_ROOT)/runtime/include -I$(SIDL_RUNTIME)/chpl
CFLAGS=`babel-config --flags-c` -std=c99
LIBS=`babel-config --libs-c-client`

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
CHPL=chpl --fast
# CHPL=chpl --print-commands --print-passes

CHPL_FLAGS=-std=c99 -DCHPL_TASKS_H=\"tasks-fifo.h\" -DCHPL_THREADS_H=\"threads-pthreads.h\" -I$(CHAPEL_ROOT)/runtime/include/tasks/fifo -I$(CHAPEL_ROOT)/runtime/include/threads/pthreads -I$(CHAPEL_ROOT)/runtime/include/comm/none -I$(CHAPEL_ROOT)/runtime/include/comp-gnu -I$(CHAPEL_ROOT)/runtime/include/$(CHPL_HOST_PLATFORM) -I$(CHAPEL_ROOT)/runtime/include -I. -Wno-all 

CHPL_LDFLAGS=-L$(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads $(CHAPEL_MAKE_SUBSTRATE_DIR)/tasks-fifo/threads-pthreads/main.o -lchpl -lm  -lpthread -lsidlstub_chpl

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

ifeq ($(IMPLSRCS),)
.chpl.lo:
	$(CHPL) --savec $<.dir $< $(IORHDRS) $(STUBHDRS) $(CHPL_HEADERS) $(DCE) --make true  # gen C-code only
	babel-libtool --mode=compile --tag=CC $(CC) \
            -I./$<.dir $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) \
            $(CHPL_FLAGS) -c -o $@ $<.dir/_main.c
else
.chpl.lo:
	$(CHPL) --savec $<.dir $< $(IORHDRS) $(STUBHDRS) $(CHPL_HEADERS) $(DCE) --make true  # gen C-code
	headerize $<.dir/_config.c $<.dir/Chapel*.c $<.dir/Default*.c $<.dir/DSIUtil.c $<.dir/chpl*.c $<.dir/List.c $<.dir/Math.c $<.dir/Search.c $<.dir/Sort.c $<.dir/Types.c
	perl -pi -e 's/((chpl__autoDestroyGlobals)|(chpl_user_main)|(chpl__init)|(chpl_main))/$*_\1/g' $<.dir/$*.c
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

