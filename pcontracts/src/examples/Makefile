ROOT_DIR=../..
include $(ROOT_DIR)/make.inc
include $(ROOT_DIR)/make_test.inc

CHECK_EXAMPLES_STATUS=$(ROOT_DIR)/$(CHECK_STATUS)-examples


FLAGS=$(BASE_FLAGS)
INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR) 

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) $(BASE_CXXFLAGS)


LKC_EXE=knapsack
UKC_EXE=unlabeledknapsack
LKCPP_EXE=Knapsack-v2
UKCPP_EXE=UnlabeledKnapsack-v2

BINARIES=$(LKC_EXE) $(UKC_EXE) $(LKCPP_EXE) $(UKCPP_EXE)


all: $(BINARIES)


.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

check: checkclean all checksetup checkKnapsackTests
	@$(MAKE) checksummary; \
	echo "DONE (check)"

checkall: check 
	@echo "DONE (checkall)"; echo

checkclean:
	rm -f *.o *.mout $(BINARIES)

checksetup: 
	@if [ ! -f $(CHECK_EXAMPLES_STATUS) ]; then \
	  echo 1 > $(CHECK_EXAMPLES_STATUS); \
	fi

checksummary:
	@echo; \
	if [ -f $(CHECK_EXAMPLES_STATUS) ]; then \
	  res=`cat $(CHECK_EXAMPLES_STATUS)`; \
	  if [ "$$res" -eq "0" ]; then \
	    echo; echo "Examples Test Suite: FAILED";\
	    echo; echo "Review output to identify failure(s)"; \
	    echo 0 > $(CHECK_ALL_STATUS); \
	  else \
	    echo; echo "Examples Test Suite: PASSED";\
	  fi; \
	  rm -f $(CHECK_EXAMPLES_STATUS); \
	else \
	  echo; echo "***********"; \
	  echo "WARNING:  Cannot assess test suite."; \
	  echo "          $(CHECK_EXAMPLES_STATUS) is missing."; \
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
	          echo "....*** Different: $$PRE $$COMP ***"; \
	          echo 0 > $(CHECK_EXAMPLES_STATUS); \
	        else \
	          echo "....Identical: $$PRE $$COMP"; \
	          if [ -f $$DIFFS ]; then \
	            echo ".......Removing empty differences file"; \
	            rm -f $$DIFFS; \
	          fi; \
	        fi; \
	      else \
	        echo "....ERROR:  Missing COMP definition"; \
	        echo 0 > $(CHECK_EXAMPLES_STATUS); \
	      fi; \
	    else \
	      echo "....ERROR:  Missing PRE definition"; \
	      echo 0 > $(CHECK_EXAMPLES_STATUS); \
	    fi; \
	  else \
	    echo "....ERROR:  Missing EXE2 definition"; \
	    echo 0 > $(CHECK_EXAMPLES_STATUS); \
	  fi; \
	else \
	  echo "....ERROR:  Missing EXE1 definition"; \
	  echo 0 > $(CHECK_EXAMPLES_STATUS); \
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

clean:
	@rm -f *.o *.out *.mout *.diffs

cleanall: clean
	@rm -f $(BINARIES) *.dot *.pdf; \
	if [ "$(RES_DIR)" != "" ]; then \
	  rm -rf $(RES_DIR)/*; \
	fi
