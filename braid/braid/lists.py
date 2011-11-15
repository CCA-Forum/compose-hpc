#!/usr/bin/env python
# -*- python -*-
## @package lists
#
# Some often use list-manipulation functions
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
import itertools

def unzip(lst):
    """
    Like zip(*lst), but works also for empty lists.
    """
    return [a for a, b in lst], [b for a, b in lst]

def unzip3(lst):
    """
    unzip for three elements
    """
    return [a for a, b, c in lst], [b for a, b, c in lst], [c for a, b, c in lst]

def flatten2d(lst):
    return list(itertools.chain.from_iterable(lst))
