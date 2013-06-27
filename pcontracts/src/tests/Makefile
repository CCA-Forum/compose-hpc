ROOT_DIR=../..

include $(ROOT_DIR)/make.inc

LIB_DIR=$(ROOT_DIR)/lib

SRC_DIR=$(ROOT_DIR)/src
RT_DIR=$(SRC_DIR)/runtime

CHECK_ALL_STATUS=$(ROOT_DIR)/$(CHECK_STATUS)

CXX=g++
FLAGS=-g -O3 
INCLUDES=-I. -I${RT_DIR}

# TBD/ToDo:  Need to resolve library linking issue
CE_OBJ=${RT_DIR}/ContractsEnforcer.o

DIR=`pwd`
LDFLAGS=$(FLAGS) -L${LIB_DIR} -lPaulContracts001a 
CXXFLAGS=$(INCLUDES) $(FLAGS) -Wno-deprecated

TCE_EXE=testContractsEnforcer
BINARIES=$(TCE_EXE)

C_FILES = $(TCE_EXE).c
C_OBJECTS = $(C_FILES:.c=.o)
CXX_FILES = 
CXX_OBJECTS = $(CXX_FILES:.cpp=.o)

all: $(BINARIES)

check: checkAll

checkAll: checkTCE

.c.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(TCE_EXE): $(TCE_EXE).o $(CE_OBJ)
	$(CXX) -o $(TCE_EXE) $(LDFLAGS) $(TCE_EXE).o $(CE_OBJ)

checkTCE: $(TCE_EXE)
	@echo "\nRunning Contracts Enforcer tests with default options..."; \
	./$(TCE_EXE) > run$(TCE_EXE).out 2>&1; \
	SUCCESS=`grep "TEST SUITE PASSED" run$(TCE_EXE).out`; \
	if [ -z "$$SUCCESS" ]; then \
	  echo "...FAILED"; \
	  echo 0 > $(CHECK_ALL_STATUS); \
	else \
	  echo "...PASSED"; \
	fi; \
	echo

clean:
	@rm -f *.o *.csv *.out

clean-all: cleanAll

cleanAll: clean
	@rm -f $(BINARIES)
