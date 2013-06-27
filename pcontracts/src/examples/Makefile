ROOT_DIR=../..

include $(ROOT_DIR)/make.inc

SRC_DIR=..

RES_DIR=$(ROOT_DIR)/results
RT_DIR=$(SRC_DIR)/runtime

BASE_RES_DIR=$(RES_DIR)/base
BASE_ORACLE_DIR=$(SRC_DIR)/data

CHECK_ALL_STATUS=$(ROOT_DIR)/$(CHECK_STATUS)
CHECK_SRC_STATUS=$(SRC_DIR)/$(CHECK_STATUS)

RI_DIR=RoutineInstrumenter
VI_DIR=VisitInstrumenter
INSTR_DIRS=$(RI_DIR) $(VI_DIR)

MAKE=make --no-print-directory
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

ROSE_OPTS = -rose:skipfinalCompileStep

ROSE_LDFLAGS = $(ROSE_LIB) $(BOOST_LIB) $(FLAGS)
ROSE_CXXFLAGS = $(ROSE_INCLUDES) $(ROSE_FLAGS) -Wno-deprecated


EXAMPLES=-name "*sack*.c*" -o -name "hello*.c*" 

LKC_EXE=knapsack
UKC_EXE=unlabeledknapsack
LKCPP_EXE=Knapsack
UKCPP_EXE=UnlabeledKnapsack
CPRINTER_EXE=ContractPrinter
CAPRINTER_EXE=ContractAssertionPrinter
RCI_EXE=RoutineContractInstrumenter
VCI_EXE=VisitContractsInstrumenter
#VCI_EXE=

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

check: checkClean all checkSetup checkKnapsackTests checkRose
	@$(MAKE) checkSummary; \
	echo "DONE (check)"; echo

checkAll: checkClean all checkSetup checkKnapsackTests checkRose checkContracts
	@$(MAKE) checkSummary; \
	echo "DONE (checkAll)"; echo

checkClean:
	rm -f *.o *.mout $(BINARIES)

checkContracts: all checkSetup
	@echo; \
	for dir in $(INSTR_DIRS); do \
	  echo; echo "Checking instrumented $$dir sources..."; \
	  CSFN=$(CHECK_ALL_STATUS)-$$dir; \
	  $(MAKE) -n -C $(RES_DIR)/$$dir checkAll || echo 0 > $$CSFN; \
	  if [ -f "$$CSFN" ]; then \
	    res=`cat $$CSFN`; \
	    if [ "$$res" -eq "0" ]; then \
	      echo "$$dir test suite failed..."; \
	      echo 0 > $(CHECK_SRC_STATUS); \
	    else \
	      echo "$$dir test suite passed..."; \
	    fi; \
	  else \
	    echo "ERROR: Unable to detect $$CSFN"; \
	    echo 0 > $(CHECK_SRC_STATUS); \
	  fi; \
	done

checkSetup: 
	@if [ ! -f $(CHECK_SRC_STATUS) ]; then \
	  echo 1 > $(CHECK_SRC_STATUS); \
	fi

checkSummary:
	@echo; \
	if [ -f $(CHECK_SRC_STATUS) ]; then \
	  res=`cat $(CHECK_SRC_STATUS)`; \
	  if [ "$$res" -eq "0" ]; then \
	    echo; echo "Complete Test Suite:  FAILED";\
	    echo; echo "Review output to identify failure(s)"; \
	    echo 0 > $(CHECK_ALL_STATUS); \
	  else \
	    echo; echo "Complete Test Suite:  PASSED";\
	  fi; \
	  rm -f $(CHECK_SRC_STATUS); \
	else \
	  echo; echo "***********"; \
	  echo "WARNING:  Cannot assess test suite."; \
	  echo "          $(CHECK_SRC_STATUS) is missing."; \
	  echo "***********"; \
	  echo 0 > $(CHECK_ALL_STATUS); \
	fi

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
	@echo "..Comparing runs of $$EXE1 and $$EXE2 with target $$KT"; \
	mkdir -p $(BASE_RES_DIR); \
	if [ "$$EXE1" != "" ]; then \
	  if [ "$$EXE2" != "" ]; then \
	    if [ "$$PRE" != "" ]; then \
	      if [ "$$COMP" != "" ]; then \
	        OFN1=$(BASE_RES_DIR)/$$EXE1-$$KT.mout; \
	        ./$$EXE1 $$KT > $$OFN1; \
	        OFN2=$(BASE_RES_DIR)/$$EXE2-$$KT.mout; \
	        ./$$EXE2 $$KT > $$OFN2; \
	        DIFFS=$(BASE_RES_DIR)/$$PRE-$$KT.diffs; \
	        diff $$OFN1 $$OFN2 > $$DIFFS; \
                if [ -s $$DIFFS ]; then \
	          echo "....*** $$PRE $$COMP are different ***"; \
	          echo 0 > $(CHECK_SRC_STATUS); \
	        else \
	          echo "....$$PRE $$COMP are identical"; \
	          if [ -f $$DIFFS ]; then \
	            echo ".......Removing empty differences file"; \
	            rm -f $$DIFFS; \
	          fi; \
	        fi; \
	      else \
	        echo "....ERROR:  Missing COMP definition"; \
	        echo 0 > $(CHECK_SRC_STATUS); \
	      fi; \
	    else \
	      echo "....ERROR:  Missing PRE definition"; \
	      echo 0 > $(CHECK_SRC_STATUS); \
	    fi; \
	  else \
	    echo "....ERROR:  Missing EXE2 definition"; \
	    echo 0 > $(CHECK_SRC_STATUS); \
	  fi; \
	else \
	  echo "....ERROR:  Missing EXE1 definition"; \
	  echo 0 > $(CHECK_SRC_STATUS); \
	fi

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
	@echo; \
	$(MAKE) checkKnapsacks KT=-1; \
	$(MAKE) checkKnapsacks KT=0; \
	$(MAKE) checkKnapsacks KT=10; \
	$(MAKE) checkKnapsacks KT=; \
	echo "DONE (checkKnapsackTests)"

checkRose: checkRosePrinters checkRoseTranslators

checkRosePrinters: checkCAPrinter checkCPrinter

checkCAPrinter: $(CAPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CAPRINTER_EXE) SDIR="CAPrinter"

checkCPrinter: $(CPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CPRINTER_EXE) SDIR="CPrinter"

checkRosePrinter: $(EXE)
	@echo; echo "Running $(EXE)..."; \
	OUTDIR=$(RES_DIR)/$$SDIR; \
	mkdir -p $$OUTDIR; \
	n=0; \
	FILES=`find ../examples $(EXAMPLES) | grep -v rose | grep -v .swp | grep -v .config | grep -v checks | grep -v diffs`; \
	for file in $$FILES; do \
	  n=`expr $$n + 1`; \
	  BASE=`echo $$file | sed 's/..\/examples\/\([^.]*\).*/\1/g'`; \
	  EXT=`echo $$file | sed 's/..\/examples\/[^.]*.\(.*\)/\1/g'`; \
	  echo "..Processing $$BASE.$$EXT"; \
	  OFN=$$OUTDIR/$$BASE.mout; \
	  ./$(EXE) $(ROSE_OPTS) -c $$file > $$OFN; \
	  if [ -s $$OFN ]; then \
	    ok=`expr $$ok + 1`; \
	    echo "....Results in $$OFN"; \
	  else \
	    echo "....FAILED on $$file"; echo; \
	    echo 0 > $(CHECK_SRC_STATUS); \
	  fi; \
	done; \
	echo "..Successfully processed $$ok of $$n files."; \
	echo "DONE (checkRosePrinter)"; echo

checkRoseTranslators: checkRCI checkVCI

#checkRoseTranslators: checkRCI 

checkRCI: $(RCI_EXE)
	@$(MAKE) checkRoseTrans EXE=$(RCI_EXE) SDIR="$(RI_DIR)"

checkVCI: $(VCI_EXE)
	@$(MAKE) checkRoseTrans EXE=$(VCI_EXE) SDIR="$(VI_DIR)"

checkRoseTrans: $(EXE)
	@echo; echo "Running $(EXE)..."; \
	OUTDIR=$(RES_DIR)/$$SDIR; \
	mkdir -p $$OUTDIR; \
	n=0; \
	FILES=`find ../examples $(EXAMPLES) | grep -v rose | grep -v .swp | grep -v .config | grep -v checks | grep -v diffs`; \
	ok=0; \
	for file in $$FILES; do \
	  n=`expr $$n + 1`; \
	  BASE=`echo $$file | sed 's/..\/examples\/\([^.]*\).*/\1/g'`; \
	  EXT=`echo $$file | sed 's/..\/examples\/[^.]*.\(.*\)/\1/g'`; \
	  echo "..Processing $$BASE.$$EXT"; \
	  OFN=rose_$$BASE.$$EXT; \
	  ERRFN=$$OUTDIR/$$BASE.err; \
	  ./$(EXE) $(ROSE_OPTS) $$file 2> $$ERRFN; \
	  ORFN=$(BASE_ORACLE_DIR)/$$SDIR/$$BASE.checks; \
	  if [ -f $$OFN ]; then \
	    ERFN=$$BASE.checks; \
	    grep PCE_CHECK $$OFN | grep ASSERT | wc -l > $$ERFN; \
	    grep PCE_CHECK $$OFN | grep INVARIANT | wc -l >> $$ERFN; \
	    grep PCE_CHECK $$OFN | grep POSTCONDITION | wc -l >> $$ERFN;\
	    grep PCE_CHECK $$OFN | grep PRECONDITION | wc -l >> $$ERFN; \
	    grep PCE_FINALIZE $$OFN | wc -l >> $$ERFN; \
	    grep PCE_INITIALIZE $$OFN | wc -l >> $$ERFN; \
	    grep PCE_UPDATE_EST_TIME $$OFN | wc -l >> $$ERFN; \
	    diff $$ERFN $$ORFN > $$ERFN.diffs; \
            if [ -s $$ERFN.diffs ]; then \
	      echo "....*** $$ERFN $$ORFN are different ***"; \
	      cat $$ERFN.diffs; \
	      echo "....FAILED on $$file"; echo; \
	      echo 0 > $(CHECK_SRC_STATUS); \
	    else \
	      echo "....$$ERFN $$ORFN are identical"; \
	      ok=`expr $$ok + 1`; \
	    fi; \
	    DESTFN=$$OUTDIR/$$BASE.$$EXT; \
	    mv -f $$OFN $$DESTFN; \
	    echo "....Unparse results in $$DESTFN"; \
	  else \
	    echo "....No output generated.  FAILED on $$file"; \
	    echo 0 > $(CHECK_SRC_STATUS); \
	  fi; \
	  if [ ! -s $$ERRFN ]; then \
	    echo "....Removing empty error file."; \
	    rm -f $$ERRFN; \
	  fi; \
	done; \
	echo; echo "..Successfully processed $$ok of $$n files."; echo; \
        if [ "$$ok" = "$$n" ]; then \
	  echo "Copying Makefile to $$OUTDIR"; \
	  cat Makefile.trans | sed 's/REPLACEME/${SDIR}/g' > $$OUTDIR/Makefile;\
	  echo "Copying configuration file(s) to $$OUTDIR"; \
	  cp -f *.config $$OUTDIR; \
	  echo "Copying header files to $$OUTDIR"; \
	  cp -f *ack*.h* $$OUTDIR; \
	  echo "Try 'make PCEFLAGS=-DPAUL_CONTRACTS check' in $$OUTDIR"; \
	else \
	  echo "ERROR: Must resolve issues before able to build translations.";\
	  echo 0 > $(CHECK_SRC_STATUS); \
        fi; \
	echo "DONE (checkRoseTran)"


clean:
	@rm -f *.o *.out *.mout *.diffs *.ti rose_* *.res *.checks

clean-all: cleanAll

cleanAll: clean
	@rm -f $(BINARIES) *.dot *.pdf; \
	if [ "$(RES_DIR)" != "" ]; then \
	  rm -rf $(RES_DIR)/*; \
	fi
