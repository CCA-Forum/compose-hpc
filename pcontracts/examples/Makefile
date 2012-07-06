MAKE = make --no-print-directory
CXX=g++
FLAGS=-g -O3
INCLUDES=-I.

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) -Wno-deprecated

ROSE_LIB = -lrose -L$(ROSE_HOME)/lib
# -Wl,-rpath $(ROSE_HOME)/lib 
BOOST_LIB = -L$(BOOST_HOME)/lib
# -Wl,-rpath $(BOOST_HOME)/lib
ROSE_INCLUDES = -I$(ROSE_HOME)/include -I$(BOOST_HOME)/include $(INCLUDES)
#ROSE_FLAGS = --edg:no_warnings
ROSE_FLAGS = $(FLAGS)
ROSE_OPTS =

ROSE_LDFLAGS = $(ROSE_LIB) $(BOOST_LIB) $(FLAGS)
ROSE_CXXFLAGS = $(ROSE_INCLUDES) $(ROSE_FLAGS) -Wno-deprecated


KC_EXE=knapsack
KCPP_EXE=Knapsack
CPRINTER_EXE=ContractPrinter

BINARIES=$(KC_EXE) $(KCPP_EXE) $(CPRINTER_EXE)


all: $(BINARIES)

check: checkKnapsacks checkCPrinter

$(KC_EXE): $(KC_EXE).c
	$(CXX) -c $(CXXFLAGS) -o $(KC_EXE).o $(KC_EXE).c
	$(CXX) -o $(KC_EXE) $(LDFLAGS) $(KC_EXE).o

$(KCPP_EXE): $(KCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(KCPP_EXE).o $(KCPP_EXE).cpp
	$(CXX) -o $(KCPP_EXE) $(LDFLAGS) $(KCPP_EXE).o

$(CPRINTER_EXE): $(CPRINTER_EXE).cpp
	$(CXX) -c $(ROSE_CXXFLAGS) -o $(CPRINTER_EXE).o $(CPRINTER_EXE).cpp
	$(CXX) -o $(CPRINTER_EXE) $(ROSE_LDFLAGS) $(CPRINTER_EXE).o

checkKnapsacks:
	@$(MAKE) checkKnapsack KT=-1; \
	$(MAKE) checkKnapsack KT=0; \
	$(MAKE) checkKnapsack KT=10; \
	$(MAKE) checkKnapsack KT=; \
	echo "\nDONE"

checkKnapsack: $(KC_EXE) $(KCPP_EXE)
	@echo "Running knapsack solvers with target=$$KT..."; \
	./Knapsack $$KT > runKcpp$$KT.out; \
	./knapsack $$KT > runKc$$KT.out; \
	diff runKcpp$$KT.out runKc$$KT.out > runK$$KT.diffs; \
        if [ -s runK$$KT.diffs ]; then \
	  echo "...Results are different"; \
	else \
	  echo "...Results are identical"; \
	fi

checkCPrinter: $(CPRINTER_EXE)
	@echo "Running $(CPRINTER_EXE)..."; \
	n=0; \
	CFILES=`find ../examples -name "*sack*.c" -print | grep -v rose | grep -v .swp`; \
	CPPFILES=`find ../examples -name "*sack*.cpp" -print | grep -v rose | grep -v .swp`; \
	FILES=`echo $$CFILES $$CPPFILES`; \
	for file in $$FILES; do \
	  n=`expr $$n + 1`; \
	  BASE=`echo $$file | sed 's/..\/examples\/\([^.]*\).*/\1/g'`; \
	  EXT=`echo $$file | sed 's/..\/examples\/[^.]*.\(.*\)/\1/g'`; \
	  echo "...Processing $$BASE.$$EXT"; \
	  ./$(CPRINTER_EXE) $(ROSE_OPTS) -c $$file > $$BASE.out; \
	  if [ -s $$BASE.out ]; then \
	    ok=`expr $$ok + 1`; \
	    echo "......Results in $$BASE.out"; \
	  else \
	    echo "......FAILED on $$file\n"; \
	  fi; \
	done; \
	echo "\n...Successfully processed $$ok of $$n files.\nDONE"


clean:
	@rm -f *.o *.out *.diffs *.ti

clean-all: clean
	@rm -f $(BINARIES) *.dot *.pdf
