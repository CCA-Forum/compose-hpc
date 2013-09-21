ROOT_DIR=../..
include $(ROOT_DIR)/src/make.inc
include $(ROOT_DIR)/src/make_test.inc


FLAGS=$(BASE_FLAGS)
INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR)

# TBD/ToDo:  Need to resolve library linking issue
CE_OBJ=$(RT_DIR)/ContractsEnforcer.o

LDFLAGS=$(FLAGS) -L$(LIB_DIR) -l$(CONTRACTS_LIB_BASE) -l$(RUNTIME_LIB_BASE)
CXXFLAGS=$(INCLUDES) $(FLAGS) $(BASE_CXXFLAGS)


TCE_EXE=testContractsEnforcer
BINARIES=$(TCE_EXE)

C_FILES = $(TCE_EXE).c
C_OBJECTS = $(C_FILES:.c=.o)
CXX_FILES = 
CXX_OBJECTS = $(CXX_FILES:.cpp=.o)

all: $(BINARIES)

.c.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(TCE_EXE): $(TCE_EXE).o $(CE_OBJ)
	$(CXX) -o $(TCE_EXE) $(LDFLAGS) $(TCE_EXE).o $(CE_OBJ)


check: checkall

checkall: checkTCE

checkTCE: $(TCE_EXE)
	@echo "\nRunning Contracts Enforcer tests with default options..."; \
	./$(TCE_EXE) > run$(TCE_EXE).out 2>&1; \
	echo; cat run$(TCE_EXE).out; echo; \
	SUCCESS=`grep "TEST SUITE PASSED" run$(TCE_EXE).out`; \
	if [ -z "$$SUCCESS" ]; then \
	  echo "Contracts Enforcer Test Suite: FAILED"; \
	  echo 0 > $(CHECK_ALL_STATUS); \
	else \
	  echo "Contracts Enforcer Test Suite: PASSED"; \
	fi; \
	echo

clean:
	@rm -f *.o *.csv *.out

cleanall: clean
	@rm -f $(BINARIES)
