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

ODIR=./results
EX_ODIR=$(ODIR)/examples
ROSE_ODIR=$(ODIR)/rose

LKC_EXE=knapsack
UKC_EXE=unlabeledknapsack
LKCPP_EXE=Knapsack
UKCPP_EXE=UnlabeledKnapsack
CPRINTER_EXE=ContractPrinter
CAPRINTER_EXE=ContractAssertionPrinter

KS_BINARIES=$(LKC_EXE) $(UKC_EXE) $(LKCPP_EXE) $(UKCPP_EXE)
ROSE_BINARIES= $(CPRINTER_EXE) $(CAPRINTER_EXE)

BINARIES=$(KS_BINARIES) $(ROSE_BINARIES)


all: $(BINARIES)

check: checkKnapsackTests checkRosePrinters

$(KC_EXE): $(KC_EXE).c
	$(CXX) -c $(CXXFLAGS) -o $(KC_EXE).o $(KC_EXE).c
	$(CXX) -o $(KC_EXE) $(LDFLAGS) $(KC_EXE).o

$(LKC_EXE): $(LKC_EXE).c
	$(CXX) -c $(CXXFLAGS) -o $(LKC_EXE).o $(LKC_EXE).c
	$(CXX) -o $(LKC_EXE) $(LDFLAGS) $(LKC_EXE).o

$(KCPP_EXE): $(KCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(KCPP_EXE).o $(KCPP_EXE).cpp
	$(CXX) -o $(KCPP_EXE) $(LDFLAGS) $(KCPP_EXE).o

$(CPRINTER_EXE): $(CPRINTER_EXE).cpp
	$(CXX) -c $(ROSE_CXXFLAGS) -o $(CPRINTER_EXE).o $(CPRINTER_EXE).cpp
	$(CXX) -o $(CPRINTER_EXE) $(ROSE_LDFLAGS) $(CPRINTER_EXE).o

$(CAPRINTER_EXE): $(CAPRINTER_EXE).cpp $(CPRINTER_EXE).cpp
	$(CXX) -c $(ROSE_CXXFLAGS) -DNO_MAIN -o $(CPRINTER_EXE)-mainLess.o  \
	  $(CPRINTER_EXE).cpp
	$(CXX) -c $(ROSE_CXXFLAGS) -o $(CAPRINTER_EXE).o $(CAPRINTER_EXE).cpp
	$(CXX) -o $(CAPRINTER_EXE) $(ROSE_LDFLAGS) $(CAPRINTER_EXE).o \
	  $(CPRINTER_EXE)-mainLess.o

checkKnapsackDiffs: 
	@echo "...Comparing runs of $$EXE1 and $$EXE2 with target $$KT..."; \
	mkdir -p $(EX_ODIR); \
	if [ "$$EXE1" != "" ]; then \
	  if [ "$$EXE2" != "" ]; then \
	    if [ "$$PRE" != "" ]; then \
	      if [ "$$COMP" != "" ]; then \
	        OFN1=$(EX_ODIR)/$$EXE1-$$KT.out; \
	        ./$$EXE1 $$KT > $$OFN1; \
	        OFN2=$(EX_ODIR)/$$EXE2-$$KT.out; \
	        ./$$EXE2 $$KT > $$OFN2; \
	        DIFFS=$(EX_ODIR)/$$PRE-$$KT.diffs; \
	        diff $$OFN1 $$OFN2 > $$DIFFS; \
                if [ -s $$DIFFS ]; then \
	          echo "......*** $$PRE $$COMP are different ***"; \
	        else \
	          echo "......$$PRE $$COMP are identical"; \
	          if [ -f $$DIFFS ]; then \
	            echo ".........Removing empty differences file"; \
	            rm -f $$DIFFS; \
	          fi; \
	        fi; \
	      else \
	        echo "......ERROR:  Missing COMP definition"; \
	      fi; \
	    else \
	      echo "......ERROR:  Missing PRE definition"; \
	    fi; \
	  else \
	    echo "......ERROR:  Missing EXE2 definition"; \
	  fi; \
	else \
	  echo "......ERROR:  Missing EXE1 definition"; \
	fi;

checkKnapsacks: $(KS_BINARIES)
	@echo; echo "Running solvers with target=$$KT..."; \
	$(MAKE) checkKnapsackDiffs EXE1=$(LKC_EXE) EXE2=$(LKCPP_EXE) \
	  PRE="Labeled" COMP="C -> C++"; \
	$(MAKE) checkKnapsackDiffs EXE1=$(UKC_EXE) EXE2=$(UKCPP_EXE) \
	  PRE="Unlabeled" COMP="C -> C++"; \
	$(MAKE) checkKnapsackDiffs EXE1=$(UKC_EXE) EXE2=$(LKC_EXE) \
	  PRE="U2L-C" COMP="Unlabeled -> Labeled"; \
	$(MAKE) checkKnapsackDiffs EXE1=$(UKCPP_EXE) EXE2=$(LKCPP_EXE) \
	  PRE="U2L-CPP" COMP="Unlabeled -> Labeled"

checkKnapsackTests:
	@$(MAKE) checkKnapsacks KT=-1; \
	$(MAKE) checkKnapsacks KT=0; \
	$(MAKE) checkKnapsacks KT=10; \
	$(MAKE) checkKnapsacks KT=; \
	echo "\nDONE"

checkRosePrinters: checkCAPrinter checkCPrinter

checkCAPrinter: $(CAPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CAPRINTER_EXE) PRE="cap"

checkCPrinter: $(CPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CPRINTER_EXE) PRE="cp"

checkRosePrinter: $(EXE)
	@echo; echo "Running $(EXE)..."; \
	mkdir -p $(ROSE_ODIR); \
	n=0; \
	CFILES=`find ../examples -name "*sack*.c" -print | grep -v rose | grep -v .swp`; \
	CPPFILES=`find ../examples -name "*sack*.cpp" -print | grep -v rose | grep -v .swp`; \
	FILES=`echo $$CFILES $$CPPFILES`; \
	for file in $$FILES; do \
	  n=`expr $$n + 1`; \
	  BASE=`echo $$file | sed 's/..\/examples\/\([^.]*\).*/\1/g'`; \
	  EXT=`echo $$file | sed 's/..\/examples\/[^.]*.\(.*\)/\1/g'`; \
	  echo "...Processing $$BASE.$$EXT"; \
	  OFN=$(ROSE_ODIR)/$$PRE-$$BASE.out; \
	  ./$(EXE) $(ROSE_OPTS) -c $$file > $$OFN; \
	  if [ -s $$OFN ]; then \
	    ok=`expr $$ok + 1`; \
	    echo "......Results in $$OFN"; \
	  else \
	    echo "......FAILED on $$file"; echo; \
	  fi; \
	done; \
	echo "\n...Successfully processed $$ok of $$n files.\nDONE"


clean:
	@rm -f *.o *.out *.diffs *.ti rose_*

clean-all: clean
	@rm -rf $(BINARIES) *.dot *.pdf $(ODIR)/*
