ROOT_DIR=../..
include $(ROOT_DIR)/src/make_rose.inc
include $(ROOT_DIR)/src/make_test.inc

CHECK_DRIVERS_STATUS=$(ROOT_DIR)/$(CHECK_STATUS)-drivers


FLAGS=$(BASE_FLAGS)
INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR)

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) $(ROSE_CXXFLAGS) $(BASE_CXXFLAGS)


# TBD/ToDo:  Need to address library linking
BASE_ROSE_SOURCES = \
	$(CONTRACTS_DIR)/RoseHelpers.cpp
BASE_ROSE_OBJECTS=$(BASE_ROSE_SOURCES:.cpp=.o)
BASE_CLAUSE_SOURCES = \
	$(BASE_ROSE_SOURCES) \
	$(CONTRACTS_DIR)/ContractsProcessor.cpp
BASE_CLAUSE_OBJECTS=$(BASE_CLAUSE_SOURCES:.cpp=.o)


CPRINTER_EXE=ContractPrinter
CAPRINTER_EXE=ContractAssertionPrinter
RCI_EXE=RoutineContractInstrumenter
VCI_EXE=VisitContractsInstrumenter

BINARIES= $(CPRINTER_EXE) $(CAPRINTER_EXE) $(RCI_EXE) $(VCI_EXE)


RI_DIR=RoutineInstrumenter
VI_DIR=VisitInstrumenter
INSTR_DIRS=$(RI_DIR) $(VI_DIR)


all: $(BINARIES)
	test -d $(BIN_DIR) || mkdir $(BIN_DIR); \
	cp -f $(BINARIES) $(BIN_DIR)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

check: checkclean all checksetup checkRose
	@$(MAKE) checksummary; \
	echo "DONE (check)"; echo

checkall: checkclean all checksetup checkRose checkcontracts
	@$(MAKE) checksummary; \
	echo "DONE (checkall)"; echo

checkclean:
	rm -f *.o *.mout $(BINARIES)

#	  (cd $$dir && make -n checkall) || echo 0 > $$CSFN; \

checkcontracts: all checksetup
	@echo; \
	for dir in $(INSTR_DIRS); do \
	  echo; echo "Checking instrumented $$dir sources..."; \
	  CSFN=$(CHECK_ALL_STATUS)-$$dir; \
	  (cd $(RES_DIR)/$$dir && make checkall) || echo 0 > $$CSFN; \
	  if [ -f "$$CSFN" ]; then \
	    res=`cat $$CSFN`; \
	    if [ "$$res" -eq "0" ]; then \
	      echo "$$dir test suite failed..."; \
	      echo 0 > $(CHECK_DRIVERS_STATUS); \
	    else \
	      echo "..$$dir test suite passed"; \
	    fi; \
	  else \
	    echo "ERROR: Unable to detect $$CSFN"; \
	    echo 0 > $(CHECK_DRIVERS_STATUS); \
	  fi; \
	done; \
	echo "DONE (checkcontracts)"; echo

checksetup: 
	@if [ ! -f $(CHECK_DRIVERS_STATUS) ]; then \
	  echo 1 > $(CHECK_DRIVERS_STATUS); \
	fi

checksummary:
	@echo; \
	if [ -f $(CHECK_DRIVERS_STATUS) ]; then \
	  res=`cat $(CHECK_DRIVERS_STATUS)`; \
	  if [ "$$res" -eq "0" ]; then \
	    echo; echo "Drivers Test Suite:  FAILED";\
	    echo; echo "Review output to identify failure(s)"; \
	    echo 0 > $(CHECK_ALL_STATUS); \
	  else \
	    echo; echo "Drivers Test Suite:  PASSED";\
	  fi; \
	  rm -f $(CHECK_DRIVERS_STATUS); \
	else \
	  echo; echo "***********"; \
	  echo "WARNING:  Cannot assess test suite."; \
	  echo "          $(CHECK_DRIVERS_STATUS) is missing."; \
	  echo "***********"; \
	  echo 0 > $(CHECK_ALL_STATUS); \
	fi


$(CPRINTER_EXE): $(CPRINTER_EXE).cpp $(BASE_ROSE_SOURCES) $(BASE_ROSE_OBJECTS)
	$(CXX) -c $(CXXFLAGS) -o $(CPRINTER_EXE).o $(CPRINTER_EXE).cpp
	$(CXX) -o $(CPRINTER_EXE) $(ROSE_LDFLAGS) $(CPRINTER_EXE).o \
	  $(BASE_ROSE_OBJECTS)

$(CAPRINTER_EXE): $(CAPRINTER_EXE).cpp $(CPRINTER_EXE).cpp \
		$(BASE_ROSE_SOURCES) $(BASE_ROSE_OBJECTS)
	$(CXX) -c $(CXXFLAGS) -DNO_MAIN -o $(CPRINTER_EXE)-mainLess.o  \
	  $(CPRINTER_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(CAPRINTER_EXE).o $(CAPRINTER_EXE).cpp
	$(CXX) -o $(CAPRINTER_EXE) $(ROSE_LDFLAGS) $(CAPRINTER_EXE).o \
	  $(CPRINTER_EXE)-mainLess.o $(BASE_ROSE_OBJECTS)

$(RCI_EXE): $(RCI_EXE).cpp $(BASE_CLAUSE_SOURCES) $(BASE_CLAUSE_OBJECTS)
	$(CXX) -c $(CXXFLAGS) -o $(RCI_EXE).o $(RCI_EXE).cpp
	$(CXX) -o $(RCI_EXE) $(ROSE_LDFLAGS) $(RCI_EXE).o $(BASE_CLAUSE_OBJECTS)

$(VCI_EXE): $(VCI_EXE).cpp $(BASE_CLAUSE_SOURCES) $(BASE_CLAUSE_OBJECTS)
	$(CXX) -c $(CXXFLAGS) -o $(VCI_EXE).o $(VCI_EXE).cpp
	$(CXX) -o $(VCI_EXE) $(ROSE_LDFLAGS) $(VCI_EXE).o $(BASE_CLAUSE_OBJECTS)


checkRose: checkRosePrinters checkRoseTranslators
	@echo; echo "DONE (checkRose)"; echo

checkRosePrinters: checkCAPrinter checkCPrinter

checkCAPrinter: $(CAPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CAPRINTER_EXE) SDIR="CAPrinter"; \
	echo "DONE ($(CAPRINTER_EXE))"

checkCPrinter: $(CPRINTER_EXE)
	@$(MAKE) checkRosePrinter EXE=$(CPRINTER_EXE) SDIR="CPrinter"; \
	echo "DONE ($(CPRINTER_EXE))"

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
	    echo 0 > $(CHECK_DRIVERS_STATUS); \
	  fi; \
	done; \
	echo "..Successfully processed $$ok of $$n files."

checkRoseTranslators: checkRCI checkVCI

checkRCI: $(RCI_EXE)
	@$(MAKE) checkRoseTrans EXE=$(RCI_EXE) SDIR="$(RI_DIR)"; \
	echo "DONE ($(RCI_EXE))"

checkVCI: $(VCI_EXE)
	@$(MAKE) checkRoseTrans EXE=$(VCI_EXE) SDIR="$(VI_DIR)"; \
	echo "DONE ($(VCI_EXE))"

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
	  ERFN=$$BASE.checks; \
	  ORFN=$(BASE_ORACLE_DIR)/$$SDIR/$$ERFN; \
	  if [ -f $$OFN ]; then \
	    grep PCE_CHECK $$OFN | grep ASSERT | wc -l > $$ERFN; \
	    grep PCE_CHECK $$OFN | grep INVARIANT | wc -l >> $$ERFN; \
	    grep PCE_CHECK $$OFN | grep POSTCONDITION | wc -l >> $$ERFN;\
	    grep PCE_CHECK $$OFN | grep PRECONDITION | wc -l >> $$ERFN; \
	    grep PCE_FINALIZE $$OFN | wc -l >> $$ERFN; \
	    grep PCE_INITIALIZE $$OFN | wc -l >> $$ERFN; \
	    grep PCE_UPDATE_EST_TIME $$OFN | wc -l >> $$ERFN; \
	    diff $$ERFN $$ORFN > $$ERFN.diffs; \
            if [ -s $$ERFN.diffs ]; then \
	      echo "....*** Different: $$ERFN $$ORFN ***"; \
	      cat $$ERFN.diffs; \
	      echo "....FAILED on $$file"; echo; \
	      echo 0 > $(CHECK_DRIVERS_STATUS); \
	    else \
	      echo "....Identical: $$ERFN $$ORFN"; \
	      ok=`expr $$ok + 1`; \
	    fi; \
	    DESTFN=$$OUTDIR/$$BASE.$$EXT; \
	    mv -f $$OFN $$DESTFN; \
	    echo "....Unparse results in $$DESTFN"; \
	  else \
	    echo "....No output generated.  FAILED on $$file"; \
	    echo 0 > $(CHECK_DRIVERS_STATUS); \
	  fi; \
	  if [ ! -s $$ERRFN ]; then \
	    echo "....Removing empty error file."; \
	    rm -f $$ERRFN; \
	  fi; \
	done; \
	echo; echo "..Successfully processed $$ok of $$n files."; echo; \
        if [ "$$ok" = "$$n" ]; then \
	  echo "Copying Makefile to $$OUTDIR"; \
	  cat ../examples/Makefile.trans | sed 's/REPLACEME/${SDIR}/g' > $$OUTDIR/Makefile;\
	  echo "Copying configuration file(s) to $$OUTDIR"; \
	  cp -f ../examples/*.config $$OUTDIR; \
	  echo "Copying header files to $$OUTDIR"; \
	  cp -f ../examples/*ack*.h* $$OUTDIR; \
	  echo "Try 'make PCEFLAGS=-DPAUL_CONTRACTS check' in $$OUTDIR"; \
	else \
	  echo "ERROR: Must resolve issues before able to build translations.";\
	  echo 0 > $(CHECK_DRIVERS_STATUS); \
        fi


clean:
	@rm -f *.o *.out *.mout *.diffs *.ti rose_* *.res *.checks

cleanall: clean
	@rm -f $(BINARIES) $(BIN_DIR)/$(BINARIES) *.dot *.pdf; \
	if [ "$(RES_DIR)" != "" ]; then \
	  rm -rf $(RES_DIR)/*; \
	fi
