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
from utils import accepts, returns, hashable
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
                  Extends, Implements, Invariants, Methods, DocComment )

        elif (sidl.interface, Name, Extends, Invariants, Methods, DocComment):
            symbol_table[Name] = \
                ( sidl.interface, (sidl.scoped_id, symbol_table.prefix, Name, ''),
                  Extends, Invariants, Methods, DocComment )

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
            build_symbol_table(UserTypes, symbol_table[sidl.Scoped_id([], Name, '')][1], verbose)

        elif (sidl.user_type, Attrs, Cipse):
            gen(Cipse)

        elif (sidl.file, Requires, Imports, UserTypes):
            gen(Imports)
            gen(UserTypes)

        elif (sidl.import_, (sidl.scoped_id, [], 'sidl', '')):
            # imported by default
            pass

        elif A:
            if (isinstance(A, list)):
                for defn in A:
                    gen(defn)
            else:
                raise Exception("build_symbol_table: NOT YET IMPLEMENTED:"+repr(A))

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
        if not sidl.is_scoped_id(scoped_id):
            return self, scoped_id

        scopes = list(scoped_id[1])+[scoped_id[2]]
        n = len(scopes)
        symbol_table = self
        # go up (and down again) in the hierarchy
        # I hope this is the expected behavior for nested packages!?
        sym = symbol_table.lookup(scopes[0])
        while not sym: # up until we find something
            symbol_table = symbol_table.parent()
            sym = symbol_table.lookup(scopes[0])

        for i in range(1, n): # down again to resolve it
            symbol_table = sym
            sym = symbol_table.lookup(scopes[i])
     
        if not sym:
            raise Exception("Symbol lookup error: "+repr(scopes))
     
        #print "successful lookup(", symbol_table, ",", scopes, ") =", sym
        return symbol_table, sym

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
                 all_names, all_methods, flags, toplevel=True):
    """
    Recursively resolve the inheritance hierarchy and build a list of
    all methods in their EPV order.
    """

    def set_method_attr(attr, (Method, Type, (MName,  Name, Extension), Attrs, Args,
                               Except, From, Requires, Ensures, DocComment)):
        return (Method, Type, (MName,  Name, Extension), [attr]+Attrs, Args,
                Except, From, Requires, Ensures, DocComment)


    def method_hash(method):
        """
        Return the long name of a method (sans class/packages)
        for sorting purposes.
        """
        def arg_hash((arg, attr, mode, typ, name)):
            if typ[0] == sidl.scoped_id:
                return mode, hashable(typ)
            if typ[0] == sidl.array:
                return hashable((mode, typ[0], sidl.hashable_type_id(typ), typ[2], typ[3]))
            return mode, typ

        return method[2], tuple([arg_hash(a) for a in sidl.method_args(method)])

    def update_method(m):
        """
        replace the element with m's name and args with m
        used to set the hooks attribute after the fact in
        case we realize the method has been overridden
        """
        (_, _, (_, name, _), _, args, _, _, _, _, _) = m

        i = 0
        for i in range(len(all_methods)):
            (_, _, (_, name1, _), _, args1, _, _, _, _, _) = all_methods[i]
            if (name1, args1) == (name, args):
                all_methods[i] = m
                return
        raise Exception('?')

    def add_method(m, with_hooks):
        if member_chk(sidl.static, sidl.method_method_attrs(m)):
            flags.has_static_methods = True

        if not method_hash(m) in all_names:
            all_names.add(method_hash(m))
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
                all_names.remove(method_hash(m))
                return
        #raise('?')

    def scan_protocols(implements):
        for impl, ifce_sym in implements:
            _, ifce = symbol_table[ifce_sym]
            if impl == sidl.implements_all:
                scan_methods(symtab, is_abstract,
                             sidl.interface_extends(ifce), 
                             [], 
                             sidl.interface_methods(ifce),
                             all_names, all_methods, flags, 
                             toplevel=False)

            for m in sidl.interface_methods(ifce):
                add_method(m, toplevel and not is_abstract)

    for _, ext in extends:
        symtab, base = symbol_table[ext]
        if base[0] == sidl.class_:
            scan_methods(symtab, is_abstract,
                         sidl.class_extends(base), 
                         sidl.class_implements(base), 
                         sidl.class_methods(base),
                         all_names, all_methods, flags, 
                         toplevel=False)
        elif base[0] == sidl.interface:
            scan_methods(symtab, is_abstract,
                         sidl.interface_extends(base), 
                         [], 
                         sidl.interface_methods(base),
                         all_names, all_methods, flags, 
                         toplevel=False)
        else: raise("?")

    scan_protocols(implements)

    for m in methods:
        if m[6]: # from clause
            remove_method(m)
        #print name, m[2], toplevel
        add_method(m, toplevel)


def visit_hierarchy(base_class, visit_func, symbol_table, visited_nodes):
    """
    Visit all parent classes and implemented interfaces of
    \c base_class exactly once and invoke visit_func on each
    sidl.class/sidl.interface node.
 
    \arg visited_nodes         An optional list of nodes
                               to exclude from visiting.
                              Contains the list of visited
                              nodes after return.
    """
 
    def step(visited_nodes, base):
 
       symtab, n = symbol_table[base]
       visit_func(symtab, n, base)
       visited_nodes.append(base)
 
       if n:
           if n[0] == sidl.class_:
               extends = n[2]
               for ext in extends:
                   if ext[1] not in visited_nodes:
                       step(visited_nodes, ext[1])
               for _, impl in n[3]:
                   if impl and impl not in visited_nodes:
                       step(visited_nodes, impl)

           elif n[0] == sidl.interface:
               for parent_interface in n[2]:
                   if parent_interface[1] and parent_interface[1] not in visited_nodes:
                       step(visited_nodes, parent_interface[1])
                           
    if base_class and base_class[1] not in visited_nodes:
       step(visited_nodes, base_class)


