#!/usr/bin/env python
# -*- python -*-
## @package sidlobjects
#
# Object-oriented representation of SIDL entities.
#
# \authors <pre>
#
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
# 
# LLNL-CODE-473891.
# All rights reserved.
#
# This file is part of BRAID ( http://compose-hpc.sourceforge.net/ ).
# Please read the COPYRIGHT file for Our Notice and for the BSD License.
#
# </pre>
#

import sidl
from sidl_symbols import visit_hierarchy, scan_methods
from patmat import member_chk
from utils import accepts, returns

@accepts(object, tuple)
def make_extendable(symbol_table, sidl_ext):
    if sidl.is_scoped_id(sidl_ext): 
        return make_extendable(*symbol_table[sidl_ext])

    if sidl.is_class(sidl_ext): 
        return Class(symbol_table, sidl_ext, [])

    if sidl.is_interface(sidl_ext): 
        return Interface(symbol_table, sidl_ext, [])

    import pdb; pdb.set_trace()


class Extendable(object):
    """
    Base class for Class, Interface, ...
    """
    def __init__(self, symbol_table, ext, attrs):
        """
        Create a new Extendable object from the sidl tuple representation.
        """
        self.data = ext
        self.is_abstract = member_chk(sidl.abstract, attrs)
        if sidl.is_scoped_id(ext[1]):
            self.symbol_table, _ = symbol_table[ext[1]]
            self.name = ext[1][2]
            self.qualified_name = ext[1][1]+[self.name+ext[1][3]]
        else:
            self.symbol_table = symbol_table
            self.name = ext[1]
            self.qualified_name = symbol_table.prefix+[self.name]

        self.extends = ext[2]
        self.implements = [] # default for interface
        self.all_parents = []

        # invoke scan_methods() to populate these
        self.all_methods = None 
        self.has_static_methods = None
        self.all_nonstatic_methods   = None 
        self.local_nonstatic_methods = None 
        self.all_static_methods      = None 
        self.local_static_methods    = None 
        self.all_nonblocking_methods = None

    def __repr__(self):
        return 'Extendable(%r, %r, %r)'%(self.symbol_table, self.data, [])

    def __str__(self):
        return self.name

    def get_scoped_id(self):
        """
        \return the scoped id of this extendable.
        """
        return sidl.Scoped_id(self.symbol_table.prefix, self.name)

    def get_parent(self):
        """
        \return the base class/interface of \c class_or_interface
        """
        extends = self.data[2]
        if extends == []:
            return None
        symbol_table, ext = self.symbol_table[extends[0][1]]

        return make_extendable(symbol_table, ext)

    def get_parents(self):
        """
        Return a list of the names of all base classes and implemented
        interfaces of a class in \c all_parents
        """
        if self.all_parents:
            return self.all_parents
        start = sidl.Scoped_id(self.symbol_table.prefix, sidl.type_id(self.data), '')
        visit_hierarchy(start, 
                        lambda _st, ext, _id: self.all_parents.append(ext), 
                        self.symbol_table, [])
        return self.all_parents

    def get_parent_interfaces(self):

        """
        return a list of the scoped ids of all implemented interfaces
        """
        isinterface = sidl.is_interface(self.data)
        isclass = sidl.is_class(self.data)
        assert isclass or isinterface

        def f(_, ext, s_id):
            if ext[0] == sidl.interface:
                # make the scoped id hashable by converting the list
                # of modules into a tuple
                sid, modules, name, ext = s_id
                all_interfaces.append((sid, tuple(modules), name, ext))

        all_interfaces = []
        #tid = sidl.type_id(self.data)
        #if not isinstance(tid, tuple):
        #    start = sidl.Scoped_id(self.symbol_table.prefix, tid, '')
        #else:
        #    start = tid
        start = self.get_scoped_id()

        visit_hierarchy(start, f, self.symbol_table, [])
        if isinterface:
            # first one is the interface itself
            return all_interfaces[1:]
        return all_interfaces

    def has_parent_interface(self, intf):
        """
        return \c True if this extendable implements the interface \c intf.
        """
        return intf in self.get_parent_interfaces()

    def get_direct_parent_interfaces(self):
        """
        return a set of all direct (local) implemented interfaces
        """

        if self.data[0] == sidl.interface:
            parents = sidl.interface_extends(self.data)
        else:
            parents = sidl.class_implements(self.data)
        return set([sidl.hashable(impl) for _, impl in parents])

    def get_unique_interfaces(self):
        """
        Extract the unique interfaces from this extendable object.
        The unique interfaces are those that belong to this class but
        do not belong to one of its parents (if they exist).  The
        returned set consists of objects of the type
        <code>Interface</code>.
        """
        unique = set(self.get_parent_interfaces())
        parent = self.get_parent()
        if parent:
            unique -= set(parent.get_parent_interfaces())
        return unique

    def get_nonstatic_methods(self, _all):
        """
        Return the list of non-static methods in this interface.
        Each element in the collection is of type <code>Method</code>.
         
        @param  all  If TRUE, then return local and parent non-static methods; 
                     otherwise, return only local non-static methods.
        """
        return self.all_nonstatic_methods if _all else self.local_nonstatic_methods


    def scan_methods(self):
        """
        Recursively resolve the inheritance hierarchy and build a list of
        all methods in their EPV order.

        This function populates the \c all_methods and the \c
        has_static_methods fields.
        """
        all_names = set()
        self.all_methods = []
        # Consolidate all methods, defined and inherited
        scan_methods(self.symbol_table, 
                     self.is_abstract, 
                     self.extends, 
                     self.implements, 
                     self.get_methods(), 
                     all_names, 
                     self.all_methods, 
                     self, 
                     toplevel=True)

        self.all_nonstatic_methods   = filter(sidl.is_not_static, self.all_methods)
        self.local_nonstatic_methods = filter(sidl.is_not_static, self.get_methods())
        self.all_static_methods      = filter(sidl.is_static, self.all_methods)
        self.local_static_methods    = filter(sidl.is_static, self.get_methods())
        # FIXME: should be all methods + send/recv for nonblocking methods
        self.all_nonblocking_methods = self.all_nonstatic_methods
        self.has_static_methods = self.all_static_methods <> []

    def number_of_methods(self):
        return len(sidl.ext_methods(self.data))

    def get_newmethods(self):
        # FIXME!!!
        return []

    def has_method_by_long_name(self, longname, _all):
        """
        Return TRUE if the specified method exists by long name; otherwise,
        return FALSE.

        @param  name  The long method name for the method to be located.
        @param  all   If TRUE then all local and parent methods are to 
                      be searched; otherwise, only local methods are to
                      be searched.
        """
        return self.lookup_method_by_long_name(longname, _all) <> None

    def lookup_method_by_long_name(self, longname, _all):
        """
        Return the <code>Method</code> with the specified long method name.  
        If there is none, return null.
    
        @param  name  The short method name for the method to be located.
        @param  all   If TRUE then all local and parent methods are to 
                      be searched; otherwise, only local methods are to
                      be searched.
        """
        for m in self.all_methods if _all else self.get_methods():
              name = sidl.method_method_name(m)
              if name[1]+name[2] == longname:
                  return m
        return None  

    def has_inv_clause(self, _all):
        """
        Returns TRUE if this Extendable has any assertions in its invariant
        clause; otherwise, returns FALSE.

        @param   all   If TRUE then check inherited invariant clauses; otherwise, 
                       only check the local clause.
        """
        if _all:
            for par in self.get_parents():
                if sidl.ext_invariants(par):
                    return True
            return False
        else:
            return sidl.ext_invariants(self.data)


    def is_interface(self):
        return sidl.is_interface(self.data)

    def is_class(self):
        return sidl.is_class(self.data)

    def get_id(self):
        """
        Same as \ref sidl.type_id()
        """
        return self.name #sidl.type_id(self.data)


    @returns(tuple)
    def get_scoped_id(self):
        """
        Same as \ref sidl.get_scoped_id()
        """
        return sidl.get_scoped_id(self.symbol_table, self.data)

class Class(Extendable):
    """
    A wrapper around the sidl class entity.
    """

    def __init__(self, symbol_table, sidl_class, attrs):
        """
        Create a new Class object from the sidl tuple representation.
        """
        super(Class, self).__init__(symbol_table, sidl_class, attrs)
        self.implements = sidl.class_implements(sidl_class)
        self.doc_comment = sidl.class_doc_comment(sidl_class)

    def is_class(self):
        return True

    def is_interface(self):
        return False

    def get_methods(self):
        return sidl.class_methods(self.data)

    def inherits_from(self, scoped_id):
        """
        \return True if this class inherits from the scoped id \c
        scoped_id or of it _is_ the class.
        """
        if self.get_scoped_id() == scoped_id:
            return True

        parent = self.get_parent()
        if parent:
            return parent.inherits_from(scoped_id)

        return False


class Interface(Extendable):
    """
    A wrapper around the sidl interface entity.
    """

    def __init__(self, symbol_table, sidl_interface, attrs):
        """
        Create a new Class object from the sidl tuple representation.
        """
        super(Interface, self).__init__(symbol_table, sidl_interface, attrs)
        self.doc_comment = sidl.interface_doc_comment(sidl_interface)

    def is_interface(self):
        return True

    def is_class(self):
        return False

    def get_methods(self):
        return sidl.interface_methods(self.data)
