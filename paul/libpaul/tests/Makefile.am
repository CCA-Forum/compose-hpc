# things autoconf will fill in
AM_CPPFLAGS = -Wall -Wno-deprecated -I$(top_srcdir)/src -I$(top_builddir)/src
rose_install  = @ROSEDIR@
boost_install = @BOOSTDIR@

# set up compiler and linker flags
AM_CPPFLAGS+= -I$(ROSEDIR)/include
AM_LDFLAGS  = -L$(ROSEDIR)/lib

AM_CPPFLAGS += -I$(BOOSTDIR)/include
AM_LDFLAGS  += -L$(BOOSTDIR)/lib

bin_PROGRAMS = test_KVAnnotation test_SXAnnotation test_PaulDecorate \
	example_traversal

example_traversal_SOURCES = example_traversal.cpp
test_KVAnnotation_SOURCES = test_KVAnnotation.cpp
test_PaulDecorate_SOURCES = test_PaulDecorate.cpp
test_SXAnnotation_SOURCES = test_SXAnnotation.cpp

example_traversal_LDADD = ../src/libpaul.la 
test_KVAnnotation_LDADD = ../src/libpaul.la 
test_PaulDecorate_LDADD = ../src/libpaul.la 
test_SXAnnotation_LDADD = ../src/libpaul.la 

EXTRA_DIST=Contracts.paulconf example.paulconf inputs

EXES=$(bin_PROGRAMS)
check-local: ${EXES}
	@echo; echo; echo "Running tests on all inputs:"; \
	INPUTS=`ls ./inputs`; \
	for exe in ${EXES}; do \
	  for file in $$INPUTS; do \
	    echo; echo "Running ./$$exe on ./inputs/$$file"; \
	    ./$$exe ./inputs/$$file; \
	    echo; \
	  done; \
	done; \
	echo "DONE"

clean:
	rm -f *.ti rose_* a.out
	rm -f *~

clean-all: clean
	rm -f ${EXES}
