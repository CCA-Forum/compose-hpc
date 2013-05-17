ifdef DEBUG
DDEBUG=-DDEBUG=$(DEBUG)
else
DDEBUG=
endif

ifdef CONFIG
DCONFIG=-DPCE_CONFIG
DDEBUG=-DDEBUG=2
else
DCONFIG=
endif

CXX=g++
FLAGS=-g -O3 ${DDEBUG} ${DCONFIG} 
INCLUDES=-I.

LIB_DIR=../../lib
LIB=$(LIB_DIR)/libPaulContracts001a.a

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) -Wno-deprecated

C_FILES=ContractsEnforcer.c

C_OBJECTS=$(C_FILES:.c=.o)

all: $(C_OBJECTS)
	@test -d $(LIB_DIR) || mkdir $(LIB_DIR); \
	ar cr $(LIB) $(C_OBJECTS)

.c.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

check:

clean:
	@rm -f *.o 

clean-all: clean
	@rm -f $(LIB)
