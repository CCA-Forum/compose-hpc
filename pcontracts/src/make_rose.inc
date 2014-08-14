#
# Dependencies:
#
# BOOST_HOME contains the path to the BOOST installation root directory
# ROOT_DIR   contains the path to the PAUL Contracts root directory
# ROSE_HOME  contains the path to the ROSE installation root directory
# JAVA_HOME  contains the path to the Java installation
# 
include $(ROOT_DIR)/src/make.inc

ROSE_LIB = -lrose -L$(ROSE_HOME)/lib
# -Wl,-rpath $(ROSE_HOME)/lib 

BOOST_LIB = -L$(BOOST_HOME)/lib
# -Wl,-rpath $(BOOST_HOME)/lib

ROSE_INCLUDES = -I$(ROSE_HOME)/include -I$(BOOST_HOME)/include $(BASE_INCLUDES)

#ROSE_FLAGS = $(BASE_FLAGS) --edg:no_warnings
ROSE_FLAGS = $(BASE_FLAGS)

ROSE_OPTS = -rose:skipfinalCompileStep
#ROSE_OPTS = 

ROSE_LDFLAGS = $(ROSE_LIB) $(BOOST_LIB) $(BASE_FLAGS)
ROSE_CXXFLAGS = $(ROSE_INCLUDES) $(ROSE_FLAGS) $(BASE_CXXFLAGS)

