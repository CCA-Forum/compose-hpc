ROOT_DIR=../..
include $(ROOT_DIR)/make.inc

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

LIB=$(LIB_DIR)/$(CONTRACTS_LIB)

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) $(BASE_CXXFLAGS)


C_FILES=ContractsEnforcer.c
C_OBJECTS=$(C_FILES:.c=.o)
C_HEADERS= \
	contractMacros.h \
	contractOptions.h \
	contractPrivateTypes.h \
	ContractsEnforcer.h \
	contracts.h


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
