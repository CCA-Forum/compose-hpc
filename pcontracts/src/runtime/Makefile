#
# Name:      src/runtime/Makefile
#
# Purpose:
# This is the PAUL Contracts runtime library makefile.  It is used to 
# manage the build and creation of the runtime library.
#
# Dependencies:
# Refer to the ../../setup.sh script for the required environment variables.


include $(PCONTRACTS_HOME)/src/make.inc


#
# DEBUG settings
ifdef DEBUG
DDEBUG=-DDEBUG=$(DEBUG)
else
DDEBUG=
endif

#Enabling configuration input file though will retain macro defs for a while.
#ifdef CONFIG
#DCONFIG=-DPCE_CONFIG
#DDEBUG=-DDEBUG=2
#else
#DCONFIG=
#endif

FLAGS=$(BASE_FLAGS) $(DDEBUG) $(DCONFIG) 
INCLUDES=$(BASE_INCLUDES)

LIB=$(LIB_DIR)/$(RUNTIME_LIB)

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) $(BASE_CXXFLAGS)


C_FILES=ContractsEnforcer.c ExpressionRoutines.c
C_OBJECTS=$(C_FILES:.c=.o)
C_HEADERS= \
	$(INCLUDE_DIR)/contractMacros.h \
	$(INCLUDE_DIR)/contractOptions.h \
	$(INCLUDE_DIR)/contractPrivateTypes.h \
	$(INCLUDE_DIR)/ContractsEnforcer.h \
	$(INCLUDE_DIR)/contracts.h \
	$(INCLUDE_DIR)/ExpressionRoutines.h

.PHONY: all check checkall clean cleanall

all: $(C_OBJECTS) $(C_HEADERS)
	test -d $(LIB_DIR) || mkdir $(LIB_DIR); \
	ar cr $(LIB) $(C_OBJECTS)

.c.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $< || echo 0 > $(PC_BUILD_STATUS)

check:

checkall:

clean:
	rm -f *.o 

cleanall: clean
	rm -f $(LIB)
