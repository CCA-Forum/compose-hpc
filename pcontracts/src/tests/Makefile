ROOT_DIR=../..
include $(ROOT_DIR)/src/make.inc
include $(ROOT_DIR)/src/make_test.inc


FLAGS=$(BASE_FLAGS)
INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR)

CE_OBJ=$(RT_DIR)/ContractsEnforcer.o

CXXFLAGS=$(INCLUDES) $(FLAGS) $(BASE_CXXFLAGS)

LDFLAGS=-L$(LIB_DIR) -l$(RUNTIME_LIB_BASE)

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
	$(CXX) -o $(TCE_EXE) $(LDFLAGS) $(TCE_EXE).o $(LDFLAGS)


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
