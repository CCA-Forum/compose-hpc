#
# Name:      src/make_rose.inc
#
# Purpose:
# This make include file defines variables required to manage the build,
# checking, and cleanup of the programs used to traverse or instrument
# C/C++ source files.
#
# Dependencies:
# Refer to the ../setup.sh script for the required environment variables.


########## You should NOT have to change anything below this line ##########

include $(PCONTRACTS_HOME)/src/make.inc

ROSE_LIB = -lrose -L$(ROSE_HOME)/lib
#ROSE_LIB = -Wl,-rpath $(ROSE_HOME)/lib 

#BOOST_LIB = -L$(BOOST_HOME)/lib
BOOST_LIB = -Wl,-rpath $(BOOST_HOME)/lib

ROSE_INCLUDES = -I$(ROSE_HOME)/include -I$(BOOST_HOME)/include $(BASE_INCLUDES)

#ROSE_FLAGS = $(BASE_FLAGS) --edg:no_warnings
ROSE_FLAGS = $(BASE_FLAGS)

ROSE_OPTS = -rose:skipfinalCompileStep -rose:collectAllCommentsAndDirectives
#ROSE_OPTS = 

ROSE_LDFLAGS = $(ROSE_LIB) $(BOOST_LIB) $(BASE_FLAGS)
ROSE_CXXFLAGS = $(ROSE_INCLUDES) $(ROSE_FLAGS) $(BASE_CXXFLAGS)

