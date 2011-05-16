#!/usr/bin/env python
# -*- python -*-
## @package sidl_parser
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
from patmat import matcher, match, member, unify, expect, Variable, unzip

def resolve(ast, verbose=True):
    """
    Run consolidate_packages(), build_symbol_table() and
    resolve_symbols() on \c ast.
    \return a tuple of the resolved ast and the symbol table.
    """
    
    ast1 = consolidate_packages(ast)
    symtab = build_symbol_table(ast1, verbose)
    ast2 = resolve_symbols(ast1, symtab, verbose)
    return ast2, symtab

@matcher(globals(), debug=False)
def build_symbol_table(node, symbol_table, verbose=True):
    """
    Does two things:

    * Build a hierarchical \c SymbolTable() for \c node.

    * Resolve all scoped ids in definitions to their full name:
      They are still stored as scoped_id, but they are guaranteed
      to contain a full name, including all parent
      namespaces/scopes.

    For the time being, we store the fully scoped name
    (= \c [package,subpackage,classname] ) for each class
    in the symbol table.
    """

    def gen(node):
        return build_symbol_table(node, symbol_table)

    with match(node):
        if (sidl.class_, Name, Extends, Implements, Invariants, Methods, DocComment):
            symbol_table[Name] = \
                ( sidl.class_, (sidl.scoped_id, symbol_table.prefix+[Name], []),
                  Extends, Implements, Invariants, Methods )

        elif (sidl.interface, Name, Extends, Invariants, Methods, DocComment):
            symbol_table[Name] = \
                ( sidl.interface, (sidl.scoped_id, symbol_table.prefix+[Name], []),
                  Extends, Invariants, Methods )

        elif (sidl.enum, Name, Items, DocComment):
            symbol_table[Name] = node

        elif (sidl.struct, (sidl.scoped_id, Names, Ext), Items, DocComment):
            symbol_table[Names[-1]] = \
                ( sidl.struct,
                  (sidl.scoped_id, symbol_table.prefix+Names, []),
                  Items )

        elif (sidl.package, Name, Version, UserTypes, DocComment):
            if (verbose):
                print "Building symbols for package %s" \
                      %'.'.join(symbol_table.prefix+[Name])

            symbol_table[Name] = SymbolTable(symbol_table,
                                             symbol_table.prefix+[Name])
            build_symbol_table(UserTypes, symbol_table[[Name]])

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
        return resolve_symbols(node, symbol_table)

    # if node <> []:
    #     print node[0], symbol_table

    with match(node):
        if (sidl.scoped_id, Names, Ext):
            return (sidl.scoped_id, symbol_table.get_full_name(Names), Ext)

        elif (sidl.package, Name, Version, UserTypes, DocComment):
            if (verbose):
                print "Resolving symbols for package %s" \
                      %'.'.join(symbol_table.prefix+[Name])
            return (sidl.package, Name, Version,
                    resolve_symbols(UserTypes, symbol_table._symbol[Name]), DocComment)

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
