#!/usr/bin/env python
# -*- python -*-
## @package parser
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
# All rights reserved.
#  
# This file is part of BRAID. For details, see 
# http://compose-hpc.sourceforge.net/. 
# Please read the COPYRIGHT file for Our Notice and
# for the BSD License.
#
# </pre>
#

import ir
from patmat import matcher, match, expect, Variable
from codegen import ClikeCodeGenerator, CCodeGenerator, SourceFile, CFile, Scope

def babel_object_type(name):
    """
    return the IR node for the type of a Babel object 'name'
    \param name    the name of the object
    """
    return (ir.pointer, (ir.struct, (ir.identifier, 's_%s__object'%name), []))

def babel_exception_type():
    """
    return the IR node for a Babel exception
    """
    return babel_object_type('sidl_BaseInterface')

class Chapel:
    class ClassInfo:
        """
        Holder object for the code generation scopes and other data
        during the traversal of the SIDL tree.
        """
        def __init__(self, impl=None, stub=None, epv=None):
            self.impl = impl
            self.stub = stub
            self.epv = epv


    def __init__(self, sidl_sexpr):
        """
        Create a new chapel code generator
        \param sidl_expr    s-expression of the SIDL data
        """
        self._sidl = sidl_sexpr

    def generate_client(self):
        """
        Generate client code.
        """
        self.generate_client1(self._sidl, None)

    @matcher(globals(), debug=False)
    def generate_client1(self, node, data):
        def gen(node):           return self.generate_client1(node, data)
        def gen1(node, data1): return self.generate_client1(node, data1)

        with match(node):
            if (ir.file_, Requires, Imports, UserTypes):
                gen(UserTypes)
            elif (ir.user_type, Attrs, Cipse):
                gen(Cipse)
            elif (ir.package, (ir.identifier, Name), Version, UserTypes):
                gen(UserTypes)
            elif (ir.class_, (ir.identifier, Name), Extends, Implements, Invariants, Methods):
                expect(data, None)
                impl = ChapelFile()
                ci = self.ClassInfo(ChapelScope(impl), CFile(), EPV(Name))
                self.gen_default_methods(Name, ci)
                gen1(Methods, ci)

                impl.new_def('module %s {'%Name)
                impl.new_def(ci.impl)
                data.impl.new_def('}')
                print Name+'.chpl:'
                print str(ci.impl)

                ci.stub.new_def((ir.var_decl, ci.epv.get_sexpr()))
                print Name+'.c:'
                print str(ci.stub)

            elif (ir.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures):
                self.generate_client_method(node, data)               
            elif A:
                if (isinstance(A, list)):
                    for defn in A:
                        gen(defn)
                else:
                    print "NOT HANDLED:", cipse
            else:
                raise Exception("match error")
        return data

    def gen_default_methods(self, name, data):
        data.epv.add_method((ir.method, ir.void, (ir.method_name, "_cast", []), [],
                             [(ir.arg, [], ir.in_, babel_object_type(name), 'self'), 
                              (ir.arg, [], ir.in_, 'const char*', 'name'), 
                              (ir.arg, [], ir.in_, babel_exception_type(), 'ex')], 
                             None, None, None, None))

    @matcher(globals(), debug=False)
    def generate_client_method(self, method, data):
        """
        Generate client code for a method interface.
        \param method   s-expression of the method's SIDL declaration
        """
        expect(method, (ir.method, _, _, _, _, _, _, _, _))
        data.epv.add_method(method)
        # output _extern declaration
        data.impl.new_def('_extern '+ chpl_gen(method))
        # output the stub definition
        stub = self.generate_stub(method, data)
        data.stub.new_header_def(c_gen(stub))


    def generate_stub(self, (Method, Type, (_, Name, _Attr), Attrs, Args, Except, From, Requires, Ensures), data):
        #return method
        expect(Method, ir.method)
        args = ([(ir.arg, [], ir.in_, babel_object_type(Name), 'self')] +
                Args +
                [(ir.arg, [], ir.in_, babel_exception_type(), 'ex')]) 
        epv_type = data.epv.get_sexpr()
        Body = (ir.stmt, 
                (ir.return_, 
                 (ir.call, 
                  (ir.get_struct_item, epv_type, (ir.deref, 'self'), 
                   'f_'+Method), args)))
        return [(ir.fn_decl, Type, Name, args),
                (ir.fn_defn, Type, Name, args, Body)]

class EPV:
    """
    Babel entry point vector for virtual method calls.
    """
    def __init__(self, name):
        self.methods = []
        self.name = name

    def add_method(self, (Method, Type, (_, Name, _Attr), Attrs, Args, Except, From, Requires, Ensures)):
        """
        add another method to the vector
        """
        self.methods.append((ir.pointer, (ir.function, Type, 'f_'+Name, Args)))
        return self

    def get_sexpr(self):
        """
        return an s-expression of the EPV declaration
        """
        return (ir.struct, (ir.identifier, 's_%s__epv'%self.name),
                [(ir.struct_item, itype, iname) for itype, iname in self.methods])

def chpl_gen(ir):
    return str(ChapelCodeGenerator().generate(ir, ChapelFile()))

def c_gen(ir):
    return str(CCodeGenerator().generate(ir, CFile()))

class ChapelFile(SourceFile):
    def __init__(self, parent=None, relative_indent=0):
        super(ChapelFile, self).__init__(
            parent, relative_indent, separator=';\n')

class ChapelScope(ChapelFile):
    def __init__(self, parent=None, relative_indent=2):
        super(ChapelScope, self).__init__(parent, relative_indent)


class ChapelCodeGenerator(ClikeCodeGenerator):
    type_map = { 
        'void':      "void",
        'bool':      "logical",
        'character': "character",
        'dcomplex':  "double complex",
        'double':    "double precision",
        'fcomplex':  "complex",
        'float':     "real",
        'int':       "int",
        'long':      "int",
        'opaque':    "int",
        'string':    "character",
        'enum':      "integer",
        'struct':    "integer",
        'class':     "integer",
        'interface': "integer",
        'package':   "void",
        'symbol':    "integer"
        }

    @matcher(globals(), debug=False)
    def generate(self, node, scope=ChapelFile()):
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            return scope.new_def(s)

        def gen_comma_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=1, separator=','))

        def gen_ws_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator=' '))

        def gen_dot_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator='.'))

        def tmap(f, l):
            return tuple(map(f, l))

        with match(node):
            if (ir.method, 'void', Name, Attrs, Args, Except, From, Requires, Ensures):
                new_def('def %s(%s)'%(gen(Name), gen_comma_sep(Args)))
            elif (ir.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures):
                new_def('def %s(%s): %s'%(gen(Name), gen_comma_sep(Args), gen(Type)))
            elif (ir.arg, Attrs, Mode, Type, Name):
                return '%s: %s'%(gen(Name), gen(Type))
            elif (ir.primitive_type, Type):
                return self.type_map[Type]
            elif (ir.attribute,   Name):    return Name
            elif (ir.identifier,  Name):    return Name
            elif (ir.method_name, Name, []): return Name
            elif (ir.method_name, Name, Extension): return Name+' '+Extension
            elif (ir.primitive_type, Name): return self.type_map[Name]
            elif (ir.scoped_id, A, B):
                return '%s%s' % (gen_dot_sep(A), gen(B))
            elif (Expr):
                return super(ChapelCodeGenerator, self).generate(Expr, scope)
            else:
                raise Exception('match error')
        return scope
