#!/usr/bin/env python
# -*- python -*-
## @package sidl_symbols
#
# Symbol table creation and symbol resolving for SIDL ASTs.
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
from utils import accepts, returns
from patmat import *
import sidl, types, sys

def resolve(ast, verbose=True):
    """
    Run consolidate_packages(), build_symbol_table() and
    resolve_symbols() on \c ast.
    \return a tuple of the resolved ast and the symbol table.
    """
    
    ast1 = consolidate_packages(ast)
    symtab = build_symbol_table(ast1, SymbolTable(), verbose)
    
    def resolve_rebuild(ast, symtab):
        """
        FIXME: this is not very efficient. Can we replace this with
        something unification-based in the future?
        """
        ast1 = resolve_symbols(ast, symtab, verbose)
        symtab1 = build_symbol_table(ast1, SymbolTable(), verbose)
        if repr(symtab) <> repr(symtab1):
            return resolve_rebuild(ast1, symtab1)
        return ast1, symtab1

    return resolve_rebuild(ast1, symtab)

@matcher(globals(), debug=False)
def build_symbol_table(node, symbol_table, verbose=True):
    """
    Build a hierarchical \c SymbolTable() for \c node.
    For the time being, we store the fully scoped name
    (= \c [package,subpackage,classname] ) for each class
    in the symbol table.
    """

    def gen(node):
        return build_symbol_table(node, symbol_table, verbose)

    with match(node):
        if (sidl.class_, Name, Extends, Implements, Invariants, Methods, DocComment):
            symbol_table[Name] = \
                ( sidl.class_, (sidl.scoped_id, symbol_table.prefix, Name, ''),
                  Extends, Implements, Invariants, Methods )

        elif (sidl.interface, Name, Extends, Invariants, Methods, DocComment):
            symbol_table[Name] = \
                ( sidl.interface, (sidl.scoped_id, symbol_table.prefix, Name, ''),
                  Extends, Invariants, Methods )

        elif (sidl.enum, Name, Items, DocComment):
            symbol_table[Name] = node

        elif (sidl.struct, Name, Items, DocComment):
            symbol_table[Name] = \
                ( sidl.struct,
                  ( sidl.scoped_id, symbol_table.prefix, Name, ''), Items, 
                  DocComment )

        elif (sidl.package, Name, Version, UserTypes, DocComment):
            if (verbose):
                import sys
                sys.stdout.write('\r'+' '*80)
                sys.stdout.write("\rBuilding symbols for package %s" 
                                 %'.'.join(symbol_table.prefix+[Name]))
                sys.stdout.flush()

            symbol_table[Name] = SymbolTable(symbol_table,
                                             symbol_table.prefix+[Name])
            build_symbol_table(UserTypes, symbol_table[sidl.Scoped_id([], Name, '')], verbose)

        elif (sidl.user_type, Attrs, Cipse):
            gen(Cipse)

        elif (sidl.file, Requires, Imports, UserTypes):
            gen(Imports)
            gen(UserTypes)

        elif A:
            if (isinstance(A, list)):
                for defn in A:
                    gen(defn)
            else:
                raise Exception("build_symbol_table: NOT HANDLED:"+repr(A))

        else:
            raise Exception("match error")
        
    return symbol_table

@matcher(globals(), debug=False)
def resolve_symbols(node, symbol_table, verbose=True):
    """
    Resolve all scoped ids in types to their full name:
    They are still stored as scoped_id, but they are guaranteed
    to contain a full name, including all parent
    namespaces/scopes.
    """

    def res(node):
        return resolve_symbols(node, symbol_table, verbose)

    #if node <> []:
    #     print node[0], symbol_table
    #if isinstance(node, tuple):
    #    if node[0] == 'class':
    #        print node[0:5]

    with match(node):
        if (sidl.scoped_id, Prefix, Name, Ext):
            prefix, name = symbol_table.get_full_name(Prefix+[Name])
            # if name == ['BaseException']:
            #     print Names, "->", prefix, name
            return (sidl.scoped_id, prefix, name, Ext)
        
        elif (sidl.package, Name, Version, UserTypes, DocComment):
            if (verbose):
                import sys
                sys.stdout.write('\r'+' '*80)
                sys.stdout.write("\rResolving symbols for package %s" 
                                 %'.'.join(symbol_table.prefix+[Name]))
                sys.stdout.flush()

            return (sidl.package, Name, Version,
                    resolve_symbols(UserTypes, symbol_table._symbol[Name],
                                    verbose), DocComment)
        else:
            if (isinstance(node, list)):
                return map(res, node)
            elif (isinstance(node, tuple)):
                return tuple(map(res, node))
            else:
                return node
            

@matcher(globals(), debug=False)
def consolidate_packages(node):
    """
    Package definitions need not be physically together in the
    SIDL file. This function sorts all packages in a user_type and
    merges adjacent packages of the same name.

    Run this function before building the symbol table!
    """

    def partition(nodes):
        packages = {}
        others = []
        for n in nodes:
            with match(n):
                if (sidl.user_type, Attr, (sidl.package, Name, Version, UTs, _)):
                    p = packages.get(Name)
                    if p: # Merge with other existing package
                        (ut, attr, (pkg, name, version, usertypes, doc_comment)) = p

                        if Version <> version:
                            print "** WARNING: Version mismatch in %s: '%s' vs. '%s'" \
                                % (Name, Version, version)

                        packages[Name] = ut, attr, (pkg, name, version,
                                                    usertypes+UTs, doc_comment)
                    else:
                        packages[Name] = n
                else:
                    others.append(n)

        return packages.values(), others 

    def cons(node): return consolidate_packages(node)

    with match(node):
        if (sidl.class_, _, _, _, _, _, _):
            return node # speedup -- cut off search

        else:
            if (isinstance(node, list)):
                pkgs, others = partition(node)
                if pkgs <> []: # also a little speedup
                    return map(cons, pkgs)+others
                else:
                    return map(cons, others)
            elif (isinstance(node, tuple)):
                return tuple(map(cons, node))
            else:
                return node


class SymbolTable(object):
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
            raise KeyError("Symbol lookup error: no parent scope")

    def lookup(self, key):
        """
        return the entry for \c key or \c None otherwise.
        """
        #print self, key, '?'
        try:
            return self._symbol[key]
        except KeyError:
            return None

    @accepts(object, tuple)
    def __getitem__(self, scoped_id):
        """
        perform a recursive symbol lookup of a scoped identifier
        """
        scopes = list(scoped_id[1])+[scoped_id[2]]
        n = len(scopes)
        symbol_table = self
        # go up (and down again) in the hierarchy
        # FIXME: Is this the expected behavior for nested packages?
        sym = symbol_table.lookup(scopes[0])
        while not sym: # up until we find something
            symbol_table = symbol_table.parent()
            sym = symbol_table.lookup(scopes[0])

        for i in range(1, n): # down again to resolve it
            sym = sym.lookup(scopes[i])
     
        if not sym:
            raise Exception("Symbol lookup error: "+repr(scopes))
     
        #print "successful lookup(", symbol_table, ",", scopes, ") =", sym
        return sym

    def __setitem__(self, key, value):
        #print self, key, '='#, value
        self._symbol[key] = value

    def get_full_name(self, scopes):
        """
        return a tuple of scopes, name for a scoped id.
        """
        found = False
        
        try:
            # go up the hierarchy until we find something
            symbol_table = self
            while not symbol_table.lookup(scopes[0]): 
                symbol_table = symbol_table.parent()
            found = True    
        except KeyError:
            pass
        
        if not found:
            # look in other symbol tables of the parent, using their prefixes
            for loop_prefix, loop_value in self.parent()._symbol.iteritems():
                if loop_prefix == self.prefix:
                    continue
                if isinstance(loop_value, SymbolTable):
                    loop_scope = scopes[0]
                    if loop_value.lookup(loop_scope):
                        symbol_table = loop_value
                        found = True
                        break
        
        if not found:
            raise Exception('Cannot resolve full name for ' + str(scopes[0]) +
                            ' from ' + str(self.prefix))
                
        #while symbol_table._parent:
        #    symbol_table = symbol_table.parent()
        #    scopes.insert(
        r = symbol_table.prefix+scopes[0:len(scopes)-1], scopes[-1]
        #print(' get_full_name: ' + str(scopes[0]) + ' resolved to ' + str(r))
        return r

    def __str__(self):
        return str(self._symbol)

    def __repr__(self):
        return repr(self._symbol)


from patmat import member_chk
import ir
def scan_methods(symbol_table, is_abstract,
                 extends, implements, methods, 
                 all_names, all_methods, flags, toplevel=False):
    """
    Recursively resolve the inheritance hierarchy and build a list of
    all methods in their EPV order.
    """

    def set_method_attr(attr, (Method, Type, (MName,  Name, Extension), Attrs, Args,
                               Except, From, Requires, Ensures, DocComment)):
        return (Method, Type, (MName,  Name, Extension), [attr]+Attrs, Args,
                Except, From, Requires, Ensures, DocComment)


    def full_method_name(method):
        """
        Return the long name of a method (sans class/packages)
        for sorting purposes.
        """
        return method[2][1]+method[2][2]

    def update_method(m):
        """
        replace the element with m's name and args with m
        used to set the hooks attribute after the fact in
        case we realize the methad has been overridden
        """
        (_, _, (_, name, _), _, args, _, _, _, _, _) = m

        i = 0
        for i in range(len(all_methods)):
            (_, _, (_, name1, _), _, args1, _, _, _, _, _) = all_methods[i]
            if (name1, args1) == (name, args):
                all_methods[i] = m
                return
        #raise ('?')

    def add_method(m, with_hooks=False):
        if member_chk(sidl.static, sidl.method_method_attrs(m)):
            flags.has_static_methods = True

        if not full_method_name(m) in all_names:
            all_names.add(full_method_name(m))
            if with_hooks:
                m = set_method_attr(ir.hooks, m)

            all_methods.append(m)
        else:
            if with_hooks:
                # override: need to update the hooks entry
                update_method(set_method_attr(ir.hooks, m))


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
                return
        #raise('?')

    def scan_protocols(implements):
        for impl in implements:
            for m in symbol_table[impl[1]][4]:
                add_method(m, toplevel and not is_abstract)

    for _, ext in extends:
        base = symbol_table[ext]
        if base[0] == sidl.class_:
            scan_methods(symbol_table, is_abstract,
                         sidl.class_extends(base), 
                         sidl.class_implements(base), 
                         sidl.class_methods(base),
                         all_names, all_methods, flags)
        elif base[0] == sidl.interface:
            scan_methods(symbol_table, is_abstract,
                         sidl.interface_extends(base), 
                         [], 
                         sidl.interface_methods(base),
                         all_names, all_methods, flags)
        else: raise("?")

    scan_protocols(implements)

    for m in methods:
        if m[6]: # from clause
            remove_method(m)
        #print name, m[2], toplevel
        add_method(m, toplevel)


def get_parents(symbol_table, class_or_interface, all_parents):
    """
    Return a list of the names of all base classes and implemented
    interfaces of a class in \c all_parents
    """

    def f(s_id):
        c_i = symbol_table[s_id]
        all_parents.append(c_i)

    start = sidl.Scoped_id(symbol_table.prefix, sidl.type_id(class_or_interface), '')
    sidl.visit_hierarchy(start, f, symbol_table, [])
    return all_parents

def get_parent_interfaces(symbol_table, class_or_interface):

    """
    return a list of the scoped ids of all implemented interfaces
    """
    isinterface = sidl.is_interface(class_or_interface)
    isclass = sidl.is_class(class_or_interface)
    assert isclass or isinterface

    def f(s_id):
        c_i = symbol_table[s_id]
        if c_i[0] == sidl.interface:
            # make it hashable
            sid, modules, name, ext = s_id
            all_interfaces.append((sid, tuple(modules), name, ext))

    all_interfaces = []
    tid = sidl.type_id(class_or_interface)
    if isclass and not isinstance(tid, tuple):
        start = sidl.Scoped_id(symbol_table.prefix, tid, '')
    else:
        start = tid

    sidl.visit_hierarchy(start, f, symbol_table, [])
    if isinterface:
        # first one is the interface itself
        return all_interfaces[1:]
    return all_interfaces


def get_direct_parent_interfaces(symbol_table, cls):

    """
    return a list of all direct (local) implemented interfaces
    """
   
    if cls[0] == sidl.interface:
        parents = sidl.interface_extends(cls)
    else:
        parents = sidl.class_implements(cls)
    return [symbol_table[impl] for _, impl in parents]
