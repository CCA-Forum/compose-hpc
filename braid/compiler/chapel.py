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
# All rights reserved.
#  
# This file is part of BRAID. For details, see 
# http://compose-hpc.sourceforge.net/. 
# Please read the COPYRIGHT file for Our Notice and
# for the BSD License.
#
# </pre>
#

import ir, sidl
from patmat import matcher, match, expect, Variable
from codegen import ClikeCodeGenerator, CCodeGenerator, SourceFile, CFile, Scope

def babel_object_type(package, name):
    """
    \return the SIDL node for the type of a Babel object 'name'
    \param name    the name of the object
    """
    return sidl.Scoped_id(package+[sidl.Id('%s__object'%name)], "")

def babel_exception_type():
    """
    \return the SIDL node for the Babel exception type
    """
    return babel_object_type([sidl.Id('sidl')], 'BaseInterface')

def ir_babel_object_type(package, name):
    """
    \return the IR node for the type of a Babel object 'name'
    \param name    the name of the object
    """
    return ir.Pointer_type(ir.Struct(babel_object_type(package,name), []))

def ir_babel_exception_type():
    """
    \return the IR node for the Babel exception type
    """
    return ir_babel_object_type([sidl.Id('sidl')], 'BaseInterface')


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
        self.sidl = sidl_sexpr

    def generate_client(self):
        """
        Generate client code. Operates in two passes:
        \li create symbol table
        \li do the work
        """
        try:       
            sym = self.build_symbol_table(self.sidl, SymbolTable())
            return self.generate_client1(self.sidl, None, sym) 
                  
        except:
            # Invoke the post-mortem debugger
            import pdb, sys
            print sys.exc_info()
            pdb.post_mortem()

    @matcher(globals(), debug=False)
    def generate_client1(self, node, data, symbol_table):
        def gen(node):         return self.generate_client1(node, data, symbol_table)
        def gen1(node, data1): return self.generate_client1(node, data1, symbol_table)

        if not symbol_table: 
            raise Exception()

        with match(node):
            if (sidl.file, Requires, Imports, UserTypes):
                gen(UserTypes)

            elif (sidl.user_type, Attrs, Cipse):
                gen(Cipse)

            elif (sidl.package, Name, Version, UserTypes):
                self.generate_client1(UserTypes, data, symbol_table[Name])

            elif (sidl.class_, (sidl.id, Name), Extends, Implements, Invariants, Methods):
                expect(data, None)
                impl = ChapelFile()
                ci = self.ClassInfo(ChapelScope(impl), CFile(), EPV(Name, symbol_table))
                self.gen_default_methods(symbol_table, Name, ci)
                gen1(Methods, ci)

                impl.new_def('module %s {'%Name)
                impl.new_def(ci.impl)
                impl.new_def('}')
                print Name+'.chpl:'
                print str(ci.impl)

                ci.stub.new_def((ir.var_decl, ci.epv.get_sexpr()))
                print Name+'.c:'
                print str(ci.stub)

            elif (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures):
                self.generate_client_method(symbol_table, node, data)               

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
    def build_symbol_table(self, node, symbol_table):
        """
        Build a hierarchical \c SymbolTable() for \c node.

        For the time being, we store the fully scoped name
        (= \c [package,subpackage,classname] ) for each class 
        in the symbol table.
        """

        def gen(node):
            return self.build_symbol_table(node, symbol_table)

        with match(node):
            if (sidl.file, Requires, Imports, UserTypes): 
                gen(UserTypes)

            elif (sidl.user_type, Attrs, Cipse): 
                gen(Cipse)

            elif (sidl.package, Name, Version, UserTypes):
                symbol_table[Name] = SymbolTable(symbol_table, 
                                                 symbol_table.prefix+[Name])
                self.build_symbol_table(UserTypes, symbol_table[Name])

            elif (sidl.class_, Name, Extends, Implements, Invariants, Methods):
                symbol_table[Name] = \
                    ( sidl.class_, (sidl.scoped_id, symbol_table.prefix+[Name], []),
                      Extends, Implements, Invariants, Methods )

            elif (sidl.struct, (sidl.scoped_id, Names, Ext), Items):
                symbol_table[Name] = \
                    ( sidl.struct, 
                      (sidl.scoped_id, symbol_table.prefix+Names, []), 
                      Items )

            elif A:
                if (isinstance(A, list)):
                    for defn in A:
                        gen(defn)
                else:
                    raise Exception("NOT HANDLED:"+repr(A))

            else:
                raise Exception("match error")

        return symbol_table

    def gen_default_methods(self, symbol_table, name, data):
        data.epv.add_method(sidl.Method(
                sidl.void,
                sidl.Method_name(sidl.Id("_cast"), ''), [],
                [sidl.Arg([], sidl.in_, babel_object_type(symbol_table.prefix, name), 
                          sidl.Id('self')), 
                 sidl.Arg([], sidl.in_, sidl.Primitive_type(sidl.string), sidl.Id('name')), 
                 sidl.Arg([], sidl.in_, babel_exception_type(), sidl.Id('ex'))],
                [], [], [], []))

    @matcher(globals(), debug=False)
    def generate_client_method(self, symbol_table, method, data):
        """
        Generate client code for a method interface.
        \param method   s-expression of the method's SIDL declaration
        """
        data.epv.add_method(method)
        # output _extern declaration
        data.impl.new_def('_extern '+ chpl_gen(method))
        # output the stub definition
        stub = self.generate_stub(symbol_table, method, data)
        data.stub.new_header_def(c_gen(stub))


    def generate_stub(self, symbol_table,
                      (Method, Type, (_,  Name, _Attr), Attrs, Args, 
                       Except, From, Requires, Ensures), data):

        def argname((_arg, _attr, _mode, _type, Id)):
            return Id
        def low(sidl_term):
            return lower_ir(symbol_table, sidl_term)

        #return method
        expect(Method, sidl.method)
        _, name = Name
        decl_args = ([ir.Arg([], ir.in_, 
                             ir_babel_object_type(symbol_table.prefix, name), 
                             ir.Id('self'))] +
                     low(Args) +
                     [ir.Arg([], ir.in_, ir_babel_exception_type(), ir.Id('ex'))]) 
        call_args = [ir.Id('self')] + map(argname, Args) + [ir.Id('ex')] 
        epv_type = data.epv.get_sexpr()
        Body = [ir.Stmt(
            ir.Return(
                ir.Call(
                    ir.Get_struct_item(
                        epv_type, 
                        ir.Deref(ir.Id('self')),
                        ir.Struct_item(
                            ir.Primitive_type(ir.void), ir.Id('f_'+Method))), 
                    call_args)))]
        return [ir.Fn_decl(low(Type), Name, decl_args),
                ir.Fn_defn(low(Type), Name, decl_args, Body)]

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
        if   (sidl.id, Name): return ir.Id(Name)
        elif (sidl.struct, Name, Items):
            return ir.Pointer_expr(ir.Struct(low_t(Name), Items))

        elif (sidl.arg, Attrs, Mode, Typ, Name): 
            return ir.Arg(Attrs, Mode, low_t(Typ), Name)

        elif (sidl.void):                 return ir.Primitive_type(ir.void)
        elif (sidl.primitive_type, Type): return low_t(sidl_term)

        elif (Terms):        
            if (isinstance(Terms, list)):
                return map(low, Terms)
        else :
            raise Exception("Not implemented")

@matcher(globals(), debug=False)
def lower_type_ir(symbol_table, sidl_type):
    """
    lower SIDL types into IR
    """
    with match(sidl_type):
        if (sidl.scoped_id, Names, Ext):
            return lower_type_ir(symbol_table, lookup_type(symbol_table, Names))
        elif (sidl.void):                        return ir.Primitive_type(ir.void)
        elif (sidl.primitive_type, sidl.opaque): return ir.Pointer_type(ir.Primitive_type(ir.void))
        elif (sidl.primitive_type, sidl.string): return ir.const_str
        elif (sidl.primitive_type, Type):        return ir.Primitive_type(Type)
        elif (sidl.class_, Name, _, _, _, _):    
            # FIXME
            return ir.Pointer_type(ir.Struct(Name, []))
        else:
            raise Exception("Not implemented")
 
def lookup_type(symbol_table, scopes):
    """
    perform a symbol lookup of a scoped identifier
    """
    n = len(scopes)
    # go up (and down again) in the hierarchy
    # FIXME: Is this the expected bahavior for nested packages?
    sym = symbol_table[scopes[0]]
    while not sym: # up until we find something
        symbol_table = symbol_table.parent()
        sym = symbol_table[scopes[0]]
        
    for i in range(0, n-1): # down again to resolve it
        sym = sym[scopes[i]]
    
    if not sym:
        raise Exception("Symbol lookup error: "+repr(key))

    #print "successful lookup(", symbol_table, ",", scopes, ") =", sym
    return sym

class SymbolTable:
    """
    Hierarchical symbol table for SIDL identifiers.
    \arg prefix  parent package. A list of identifiers 
                 just as they would appear in a \c Scoped_id()
    """
    def __init__(self, parent=None, prefix=[]):
        #print "new scope", self, 'parent =', parent
        self._parent = parent
        self._symbol = {}
        self.prefix = prefix

    def parent(self):
        if self._parent: 
            return self._parent
        else: 
            raise Exception("Symbol lookup error: no parent scope")

    @matcher(globals())
    def __getitem__(self, key):
        expect(key, (sidl.id, _))
        #print self, key, '?'
        try:
            return self._symbol[key]
        except KeyError:
            return None

    @matcher(globals())
    def __setitem__(self, key, value):
        expect(key, (sidl.id, _))
        #print self, key, '='#, value
        self._symbol[key] = value

class EPV:
    """
    Babel entry point vector for virtual method calls.
    """
    def __init__(self, name, symbol_table):
        self.methods = []
        self.name = name
        self.symbol_table = symbol_table

    def add_method(self, method):
        """
        add another (SIDL) method to the vector
        """
        def to_fn_decl((_sidl_method, Type, 
                        (Method_name, Name, Extension), 
                        Attrs, Args, Except, From, Requires, Ensures)):
            return ir.Fn_decl(lower_ir(self.symbol_table, Type), Name, Args)

        self.methods.append(to_fn_decl(method))
        return self

    def get_sexpr(self):
        """
        return an s-expression of the EPV declaration
        """
        def get_type_name((_fn_decl, Type, Name, _Args)):
            return Type, Name

        name = ir.Scoped_id(self.symbol_table.prefix+[ir.Id('%s__epv'%self.name)], '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname) 
             for itype, iname in map(get_type_name, self.methods)])

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
            if (sidl.method, 'void', Name, Attrs, Args, Except, From, Requires, Ensures):
                new_def('def %s(%s)'%(gen(Name), gen_comma_sep(Args)))

            elif (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures):
                new_def('def %s(%s): %s'%(gen(Name), gen_comma_sep(Args), gen(Type)))

            elif (sidl.arg, Attrs, Mode, Type, Name):
                return '%s: %s'%(gen(Name), gen(Type))

            elif (sidl.primitive_type, Type):         return self.type_map[Type]
            elif (sidl.custom_attribute, Name):       return Name
            elif (sidl.id,  Name):                    return Name
            elif (sidl.method_name, Name, []):        return Name
            elif (sidl.method_name, Name, Extension): return Name+' '+Extension
            elif (sidl.scoped_id, A, B):
                return '%s%s' % (gen_dot_sep(A), gen(B))

            elif (Expr):
                return super(ChapelCodeGenerator, self).generate(Expr, scope)

            else:
                raise Exception('match error')
        return scope
