#
# Name:      src/tests/Makefile
#
# Purpose:
# This is the PAUL Contracts runtime library tests makefile.  It is used to
# manage the building, checking, and cleaning up of the built-in tests for
# the runtime library.
#
# Dependencies:
# Refer to the ../../setup.sh script for the required environment variables.


include $(PCONTRACTS_HOME)/src/make_test.inc


FLAGS=$(BASE_FLAGS)
INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR)

CE_OBJ=$(RT_DIR)/ContractsEnforcer.o

CXXFLAGS=$(INCLUDES) $(FLAGS) $(BASE_CXXFLAGS)

LDFLAGS=-L$(LIB_DIR) -l$(RUNTIME_LIB_BASE)

TCE_EXE=testContractsEnforcer
TER_EXE=testExpressionRoutines
BINARIES=$(TCE_EXE) $(TER_EXE)

C_FILES = $(TCE_EXE).c
C_OBJECTS = $(C_FILES:.c=.o)
CXX_FILES = 
CXX_OBJECTS = $(CXX_FILES:.cpp=.o)

.PHONY: all check checkall checkTCE checkTER clean cleanall

all: $(BINARIES)

.c.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $< || echo 0 > $(PC_BUILD_STATUS)

$(TCE_EXE): $(TCE_EXE).o $(CE_OBJ)
	$(CXX) -o $(TCE_EXE) $(LDFLAGS) $(TCE_EXE).o $(LDFLAGS) \
	  || echo 0 > $(PC_BUILD_STATUS)

$(TER_EXE): $(TER_EXE).o $(CE_OBJ)
	$(CXX) -o $(TER_EXE) $(LDFLAGS) $(TER_EXE).o $(LDFLAGS) \
	  || echo 0 > $(PC_BUILD_STATUS)


check: checkall

checkall: checkTCE checkTER

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

checkTER: $(TER_EXE)
	@echo "\nRunning Contract Expressions tests with default options..."; \
	./$(TER_EXE) > run$(TER_EXE).out 2>&1; \
	echo; cat run$(TER_EXE).out; echo; \
	SUCCESS=`grep "TEST SUITE PASSED" run$(TER_EXE).out`; \
	if [ -z "$$SUCCESS" ]; then \
	  echo "Contracts Expressions Test Suite: FAILED"; \
	  echo 0 > $(CHECK_ALL_STATUS); \
	else \
	  echo "Contracts Expressions Test Suite: PASSED"; \
	fi; \
	echo

clean:
	@rm -f *.o *.csv *.out

cleanall: clean
	@rm -f $(BINARIES)
