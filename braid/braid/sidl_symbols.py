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
from codegen import accepts, returns
from patmat import matcher, match, member, unify, expect, Variable
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
                ( sidl.class_, (sidl.scoped_id, symbol_table.prefix, Name, []),
                  Extends, Implements, Invariants, Methods )

        elif (sidl.interface, Name, Extends, Invariants, Methods, DocComment):
            symbol_table[Name] = \
                ( sidl.interface, (sidl.scoped_id, symbol_table.prefix, Name, []),
                  Extends, Invariants, Methods )

        elif (sidl.enum, Name, Items, DocComment):
            symbol_table[Name] = node

        elif (sidl.struct, (sidl.scoped_id, Prefix, Name, Ext), Items, DocComment):
            symbol_table[Names[-1]] = \
                ( sidl.struct,
                  (sidl.scoped_id, symbol_table.prefix+Prefix, Name, Ext),
                  Items )

        elif (sidl.package, Name, Version, UserTypes, DocComment):
            if (verbose):
                print "Building symbols for package %s" \
                      %'.'.join(symbol_table.prefix+[Name])

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
                print "Building symbols for package %s" \
                      %'.'.join(symbol_table.prefix+[Name])
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
                            print "** WARNING: Version mismatch in %s:"\
                                  "'%s' vs. '%s'"% Name, Version, version

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
            raise Exception("Symbol lookup error: no parent scope")

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
        scopes = scoped_id[1]+[scoped_id[2]]
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
        except Exception:
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
            raise Exception('Cannot resolve full name for ' + str(scopes[0]) + ' from ' + str(self.prefix))
                
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
