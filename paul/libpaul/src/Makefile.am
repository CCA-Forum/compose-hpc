##
##
##
# set up compiler and linker flags
AM_CPPFLAGS = -Wall -Wno-deprecated

AM_CPPFLAGS += -I$(ROSEDIR)/include
AM_LDFLAGS = -L$(ROSEDIR)/lib

AM_CPPFLAGS += -I$(BOOSTDIR)/include
AM_LDFLAGS += -L$(BOOSTDIR)/lib

##
##
##
# 
lib_LTLIBRARIES = libpaul.la
libpaul_la_SOURCES = \
	$(BUILT_SOURCES) scanner.l \
        KVAnnotationValue.cpp ssexpr.cpp SXAnnotationValue.cpp Dynamic.cpp \
        PlainAnnotationValue.cpp PaulDecorate.cpp PaulConfReader.cpp \
        Annotation.h KVAnnotationValue.h ssexpr.h SXAnnotationValue.h \
	PaulConfReader.h PaulDecorate.h Dynamic.h PlainAnnotationValue.h \
	RoseUtilities.hpp RoseUtilities.cpp Utilities.h Utilities.cpp

EXTRA_DIST=lempar.c parser.y parser.c parser.h
BUILT_SOURCES=lemon-parser.cpp parser.h scanner.h
noinst_PROGRAMS=lemon
lemon_SOURCES = lemon.c
CLEANFILES = $(BUILT_SOURCES) parser.c
libpaul_la_LIBADD= -lrose -lfl          # Flex!

# Build parser & scanner
scanner.c scanner.h: scanner.l
	$(LEX) $<

parser.c parser.h : parser.y lemon # copying is needed for vpath builds, lemon is not flexible enough
	$(AM_V_GEN) if test ! -f parser.y; then \
          cp $< $(srcdir)/lempar.c .; \
	  ./lemon -q parser.y; \
	  rm -f lempar.c parser.y; \
        else ./lemon -q $<; fi

lemon-parser.cpp: # this way it also works with make -j
	$(AM_V_GEN) echo '#include "parser.c"' >$@
