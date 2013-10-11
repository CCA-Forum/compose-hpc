ROOT_DIR=../..
include $(ROOT_DIR)/src/make.inc

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


all: $(C_OBJECTS) $(C_HEADERS)
	test -d $(LIB_DIR) || mkdir $(LIB_DIR); \
	ar cr $(LIB) $(C_OBJECTS)

.c.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

check:

checkall:

clean:
	rm -f *.o 

cleanall: clean
	rm -f $(LIB)
