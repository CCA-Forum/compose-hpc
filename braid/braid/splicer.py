#!/usr/bin/env python
# -*- python -*-
## @package splicer
# Support functions for handling Babel Splicer blocks
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
import os,re

def replace(filename, splicer_name, text):
    """
    Replace the contents of a \c splicer_block with \c text.
    \param filename      The name of the file to edit.
    \param splicer_name  The name of the splicer block to edit.
    \param text          The text to put inside the splicer block.

    expect(string, string, string)
    """
    #print "splicing", splicer_name, "with '"+text+"'"
    if text == None: raise Exception("Empty splicer text")

    # first make a backup of the old file
    os.rename(filename, filename+'~')
    dest = open(filename, 'w')
    src = open(filename+'~', 'r')

    inside = False
    did_replace = False
    for line in src:
        if re.match(r'.*DO-NOT-DELETE splicer\.begin\('+splicer_name+r'\).*', 
                    line):
            dest.write(line)
            dest.write(text+'\n')
            inside = True
            did_replace = True
        elif (inside and
            re.match(r'.*DO-NOT-DELETE splicer\.end\('+splicer_name+r'\).*', 
                     line)):
                inside = False

        if not inside:
            dest.write(line)
                
    if inside:
        raise Exception("unclosed splicer block")

    if not did_replace:
        raise Exception("splicer block not found")

    src.close()
    dest.close()


def record(filename):
    """
    Return a dict with the contents of all splicer blocks in the file
    \c filename.
    """
    src = open(filename, 'r')
    inside = False
    splicer_name = ''
    splicers = {}
    for line in src:
        m = re.match(r'.*DO-NOT-DELETE splicer\.begin\((.*)\).*', line)
        if m:
            splicer_name = m.group(1)
            # print "splicer_block(%s)", splicer_name
            splicer = []
            inside = True
        elif (inside and re.match(
                r'.*DO-NOT-DELETE splicer\.end\(%s\).*'%splicer_name, line)):
            inside = False
            splicers[splicer_name] = splicer
        elif inside:
            splicer.extend(line)

    if inside:
        raise Exception("unclosed splicer block: "+splicer+name)
    
    src.close()
    return splicers

def apply_all(filename, splicers):
    """
    Apply the previously recorded splicers \c splicers to the file \c
    filename.
    """
    # first make a backup of the old file
    os.rename(filename, filename+'~')
    dest = open(filename, 'w')
    src = open(filename+'~', 'r')

    all_splicers = set()
    for s in splicers:
        all_splicers.add(s)

    splicer_name = ''
    inside = False
    did_replace = False
    for line in src:
        m = re.match(r'.*DO-NOT-DELETE splicer\.begin\((.*)\).*', line)
        if m:
            splicer_name = m.group(1)
            block = splicers[splicer_name]
            dest.write(line)
            for l in block: 
                dest.write(l)
            inside = True
            did_replace = True
        elif (inside and re.match(
                r'.*DO-NOT-DELETE splicer\.end\(%s\).*'%splicer_name, line)):
            inside = False
            all_splicers.remove(splicer_name)

        if not inside:
            dest.write(line)                
                
    if inside:
        raise Exception("unclosed splicer block: "+splicer_name)

    if not did_replace:
        raise Exception("splicer block not found")

    if len(all_splicers) > 0:
        raise Exception("The following splicer blocks were not found: "
                        +str(all_splicers))

    src.close()
    dest.close()
