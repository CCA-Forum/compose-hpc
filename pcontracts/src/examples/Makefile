RT_DIR=../runtime
RES_DIR=../../results
BASE_RES_DIR=$(RES_DIR)/base
ROSE_RES_DIR=$(RES_DIR)/rose

MAKE = make --no-print-directory
CXX=g++
#FLAGS=-g -O3 -DDEBUG
FLAGS=-g -O3
INCLUDES=-I. -I${RT_DIR}

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


LKC_EXE=knapsack
UKC_EXE=unlabeledknapsack
LKCPP_EXE=Knapsack
UKCPP_EXE=UnlabeledKnapsack
CPRINTER_EXE=ContractPrinter
CAPRINTER_EXE=ContractAssertionPrinter
RCI_EXE=RoutineContractInstrumenter
VCI_EXE=VisitContractInstrumenter

BASE_ROSE_SOURCES = \
	RoseHelpers.cpp
BASE_ROSE_OBJECTS=$(BASE_ROSE_SOURCES:.cpp=.o)
BASE_CLAUSE_SOURCES = \
	$(BASE_ROSE_SOURCES) \
	ContractsProcessor.cpp
BASE_CLAUSE_OBJECTS=$(BASE_CLAUSE_SOURCES:.cpp=.o)

KS_BINARIES=$(LKC_EXE) $(UKC_EXE) $(LKCPP_EXE) $(UKCPP_EXE)
ROSE_BINARIES= $(CPRINTER_EXE) $(CAPRINTER_EXE) $(RCI_EXE) $(VCI_EXE)

BINARIES=$(KS_BINARIES) $(ROSE_BINARIES)

# WARNING:  The following assumes unnamed C++ targets are associated
#  with ROSE objects.

.cpp.o:
	$(CXX) -c $(ROSE_CXXFLAGS) -o $@ $<

all: $(BINARIES)

check: checkKnapsackTests checkRose

$(KC_EXE): $(KC_EXE).c
	$(CXX) -c $(CXXFLAGS) -o $(KC_EXE).o $(KC_EXE).c
	$(CXX) -o $(KC_EXE) $(LDFLAGS) $(KC_EXE).o

$(LKC_EXE): $(LKC_EXE).c
	$(CXX) -c $(CXXFLAGS) -o $(LKC_EXE).o $(LKC_EXE).c
	$(CXX) -o $(LKC_EXE) $(LDFLAGS) $(LKC_EXE).o

$(KCPP_EXE): $(KCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(KCPP_EXE).o $(KCPP_EXE).cpp
	$(CXX) -o $(KCPP_EXE) $(LDFLAGS) $(KCPP_EXE).o

$(UKCPP_EXE): $(UKCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(UKCPP_EXE).o $(UKCPP_EXE).cpp
	$(CXX) -o $(UKCPP_EXE) $(LDFLAGS) $(UKCPP_EXE).o

$(CPRINTER_EXE): $(CPRINTER_EXE).cpp $(BASE_ROSE_SOURCES) $(BASE_ROSE_OBJECTS)
	$(CXX) -c $(ROSE_CXXFLAGS) -o $(CPRINTER_EXE).o $(CPRINTER_EXE).cpp
	$(CXX) -o $(CPRINTER_EXE) $(ROSE_LDFLAGS) $(CPRINTER_EXE).o \
	  $(BASE_ROSE_OBJECTS)

$(CAPRINTER_EXE): $(CAPRINTER_EXE).cpp $(CPRINTER_EXE).cpp \
		$(BASE_ROSE_SOURCES) $(BASE_ROSE_OBJECTS)
	$(CXX) -c $(ROSE_CXXFLAGS) -DNO_MAIN -o $(CPRINTER_EXE)-mainLess.o  \
	  $(CPRINTER_EXE).cpp
	$(CXX) -c $(ROSE_CXXFLAGS) -o $(CAPRINTER_EXE).o $(CAPRINTER_EXE).cpp
	$(CXX) -o $(CAPRINTER_EXE) $(ROSE_LDFLAGS) $(CAPRINTER_EXE).o \
	  $(CPRINTER_EXE)-mainLess.o $(BASE_ROSE_OBJECTS)

$(RCI_EXE): $(RCI_EXE).cpp $(BASE_CLAUSE_SOURCES) $(BASE_CLAUSE_OBJECTS)
	$(CXX) -c $(ROSE_CXXFLAGS) -o $(RCI_EXE).o $(RCI_EXE).cpp
	$(CXX) -o $(RCI_EXE) $(ROSE_LDFLAGS) $(RCI_EXE).o $(BASE_CLAUSE_OBJECTS)

$(VCI_EXE): $(VCI_EXE).cpp $(BASE_CLAUSE_SOURCES) $(BASE_CLAUSE_OBJECTS)
	$(CXX) -c $(ROSE_CXXFLAGS) -o $(VCI_EXE).o $(VCI_EXE).cpp
	$(CXX) -o $(VCI_EXE) $(ROSE_LDFLAGS) $(VCI_EXE).o $(BASE_CLAUSE_OBJECTS)


checkKnapsackDiffs: 
	@echo "...Comparing runs of $$EXE1 and $$EXE2 with target $$KT..."; \
	mkdir -p $(BASE_RES_DIR); \
	if [ "$$EXE1" != "" ]; then \
	  if [ "$$EXE2" != "" ]; then \
	    if [ "$$PRE" != "" ]; then \
	      if [ "$$COMP" != "" ]; then \
	        OFN1=$(BASE_RES_DIR)/$$EXE1-$$KT.out; \
	        ./$$EXE1 $$KT > $$OFN1; \
	        OFN2=$(BASE_RES_DIR)/$$EXE2-$$KT.out; \
	        ./$$EXE2 $$KT > $$OFN2; \
	        DIFFS=$(BASE_RES_DIR)/$$PRE-$$KT.diffs; \
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

checkRose: checkRosePrinters checkRoseTranslators

checkRosePrinters: checkCAPrinter checkCPrinter

checkCAPrinter: $(CAPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CAPRINTER_EXE) SDIR="cap"

checkCPrinter: $(CPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CPRINTER_EXE) SDIR="cp"

checkRosePrinter: $(EXE)
	@echo; echo "Running $(EXE)..."; \
	OUTDIR=$(ROSE_RES_DIR)/$$SDIR; \
	mkdir -p $$OUTDIR; \
	n=0; \
	CFILES=`find ../examples -name "*sack*.c" -print | grep -v rose | grep -v .swp`; \
	CPPFILES=`find ../examples -name "*sack*.cpp" -print | grep -v rose | grep -v .swp`; \
	FILES=`echo $$CFILES $$CPPFILES`; \
	for file in $$FILES; do \
	  n=`expr $$n + 1`; \
	  BASE=`echo $$file | sed 's/..\/examples\/\([^.]*\).*/\1/g'`; \
	  EXT=`echo $$file | sed 's/..\/examples\/[^.]*.\(.*\)/\1/g'`; \
	  echo "...Processing $$BASE.$$EXT"; \
	  OFN=$$OUTDIR/$$BASE.out; \
	  ./$(EXE) $(ROSE_OPTS) -c $$file > $$OFN; \
	  if [ -s $$OFN ]; then \
	    ok=`expr $$ok + 1`; \
	    echo "......Results in $$OFN"; \
	  else \
	    echo "......FAILED on $$file"; echo; \
	  fi; \
	done; \
	echo "\n...Successfully processed $$ok of $$n files.\nDONE"


#checkRoseTranslators: checkRCI checkVCI

checkRoseTranslators: checkRCI 

checkRCI: $(RCI_EXE)
	@$(MAKE) checkRoseTrans EXE=$(RCI_EXE) SDIR="rci"

checkVCI: $(VCI_EXE)
	@$(MAKE) checkRoseTrans EXE=$(VCI_EXE) SDIR="vci"

checkRoseTrans: $(EXE)
	@echo; echo "Running $(EXE)..."; \
	OUTDIR=$(ROSE_RES_DIR)/$$SDIR; \
	mkdir -p $$OUTDIR; \
	n=0; \
	CFILES=`find ../examples -name "*sack*.c" -print | grep -v rose | grep -v .swp`; \
	CPPFILES=`find ../examples -name "*sack*.cpp" -print | grep -v rose | grep -v .swp`; \
	FILES=`echo $$CFILES $$CPPFILES`; \
	ok=0; \
	for file in $$FILES; do \
	  n=`expr $$n + 1`; \
	  BASE=`echo $$file | sed 's/..\/examples\/\([^.]*\).*/\1/g'`; \
	  EXT=`echo $$file | sed 's/..\/examples\/[^.]*.\(.*\)/\1/g'`; \
	  echo "...Processing $$BASE.$$EXT"; \
	  OFN=rose_$$BASE.$$EXT; \
	  ERRFN=$$OUTDIR/$$BASE.err; \
	  ./$(EXE) $(ROSE_OPTS) $$file 2> $$ERRFN; \
	  if [ -f $$OFN ]; then \
	    ok=`expr $$ok + 1`; \
	    DESTFN=$$OUTDIR/$$BASE.$$EXT; \
	    mv -f $$OFN $$DESTFN; \
	    echo "......Unparse results in $$DESTFN"; \
	  else \
	    echo "......No output generated.  FAILED on $$file"; \
	  fi; \
	  if [ ! -s $$ERRFN ]; then \
	    echo "......Removing empty error file."; \
	    rm -f $$ERRFN; \
	  fi; \
	  echo; \
	done; \
	echo "\n...Successfully processed $$ok of $$n files.\n"; \
        if [ "$$ok" = "$$n" ]; then \
	  echo "Copying Makefile to $$OUTDIR"; \
	  cp -f Makefile.trans $$OUTDIR/Makefile; \
	  echo "Copying header files to $$OUTDIR"; \
	  cp -f *ack*.h* $$OUTDIR; \
	  echo "Try 'make PCEFLAGS=-DPAUL_CONTRACTS check' in $$OUTDIR"; \
	else \
	  echo "ERROR: Must resolve issues before able to build translations.";\
        fi; \
	echo "DONE"


clean:
	@rm -f *.o *.out *.diffs *.ti rose_*

clean-all: clean
	@rm -rf $(BINARIES) *.dot *.pdf $(RES_DIR)/*
