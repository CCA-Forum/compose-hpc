ROOT_DIR=../..
include $(ROOT_DIR)/src/make.inc
include $(ROOT_DIR)/src/make_test.inc

ORACLE_DIR=$(BASE_ORACLE_DIR)/examples
CHECK_EXAMPLES_STATUS=$(ROOT_DIR)/$(CHECK_STATUS)-examples


FLAGS=$(BASE_FLAGS)
#INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR) 
INCLUDES=$(BASE_INCLUDES)

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) $(BASE_CXXFLAGS)


LKC_EXE=knapsack
UKC_EXE=unlabeledknapsack
LKCPP_EXE=Knapsack-v2
UKCPP_EXE=UnlabeledKnapsack-v2
WLCPP_EXE=WeightsList
HW_EXE=helloworld
HW2_EXE=helloworld-v2
HW3_EXE=helloworld-v3

BINARIES=$(LKC_EXE) $(UKC_EXE) $(LKCPP_EXE) $(UKCPP_EXE) \
	$(HW_EXE) $(HW2_EXE) $(HW3_EXE) $(WLCPP_EXE)


all: $(BINARIES)


.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $< || echo 0 > $(PC_BUILD_STATUS)

.cc.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $< || echo 0 > $(PC_BUILD_STATUS)

check: checkclean all checksetup checkExamples checkKnapsackTests
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
	    echo; echo "Examples Test Suites: FAILED";\
	    echo; echo "Review output to identify failure(s)"; \
	    echo 0 > $(CHECK_ALL_STATUS); \
	  else \
	    echo; echo "Examples Test Suites: PASSED";\
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
	$(CXX) -c $(CXXFLAGS) -o $(KC_EXE).o $(KC_EXE).c \
	  || echo 0 > $(PC_BUILD_STATUS)
	$(CXX) -o $(KC_EXE) $(LDFLAGS) $(KC_EXE).o \
	  || echo 0 > $(PC_BUILD_STATUS)

$(LKC_EXE): $(LKC_EXE).c
	$(CXX) -c $(CXXFLAGS) -o $(LKC_EXE).o $(LKC_EXE).c \
	  || echo 0 > $(PC_BUILD_STATUS)
	$(CXX) -o $(LKC_EXE) $(LDFLAGS) $(LKC_EXE).o \
	  || echo 0 > $(PC_BUILD_STATUS)

$(KCPP_EXE): $(KCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(KCPP_EXE).o $(KCPP_EXE).cpp \
	  || echo 0 > $(PC_BUILD_STATUS)
	$(CXX) -o $(KCPP_EXE) $(LDFLAGS) $(KCPP_EXE).o \
	  || echo 0 > $(PC_BUILD_STATUS)

$(UKCPP_EXE): $(UKCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(UKCPP_EXE).o $(UKCPP_EXE).cpp \
	  || echo 0 > $(PC_BUILD_STATUS)
	$(CXX) -o $(UKCPP_EXE) $(LDFLAGS) $(UKCPP_EXE).o \
	  || echo 0 > $(PC_BUILD_STATUS)

$(WLCPP_EXE): $(WLCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(WLCPP_EXE).o $(WLCPP_EXE).cpp \
	  || echo 0 > $(PC_BUILD_STATUS)
	$(CXX) -o $(WLCPP_EXE) $(LDFLAGS) $(WLCPP_EXE).o \
	  || echo 0 > $(PC_BUILD_STATUS)


# TODO: Consider deprecating the labeled versus unlabeled knapsack checks.

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
	          echo "....FAILED: $$EXE1 vs $$EXE2"; \
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

# TODO: Consider deprecating the labeled versus unlabeled knapsack checks.

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

# TODO: Consider deprecating the labeled versus unlabeled knapsack checks.

checkKnapsackTests:
	@echo; \
	$(MAKE) checkKnapsacks KT=-1; \
	$(MAKE) checkKnapsacks KT=0; \
	$(MAKE) checkKnapsacks KT=10; \
	$(MAKE) checkKnapsacks KT=; \
	echo "DONE (checkKnapsackTests)"

checkExamples: $(BINARIES)
	@echo; echo "Comparing example test versus oracle results.."; \
	ok=0; \
	n=0; \
	for exe in $(BINARIES); do \
	  n=`expr $$n + 1`; \
	  echo; echo "..Processing $$exe"; \
	  if [ -f $$exe ]; then \
	    TFN=$$exe.mout; \
	    ORFN=$(ORACLE_DIR)/$$TFN; \
	    ./$$exe > $$TFN; \
	    if [ -f $$ORFN ]; then \
	      diff $$TFN $$ORFN > $$TFN.diffs; \
	      if [ -s $$TFN.diffs ]; then \
	        echo "....*** Different: $$TFN $$ORFN ***"; \
	        cat $$TFN.diffs; \
	        echo "....FAILED on $$exe"; echo; \
	        echo 0 > $(CHECK_EXAMPLES_STATUS); \
	      else \
	        echo "....Identical: $$TFN $$ORFN"; \
	        ok=`expr $$ok + 1`; \
	      fi; \
	    else \
	      echo "....*** $$ORFN is missing ***"; \
	      echo 0 > $(CHECK_EXAMPLES_STATUS); \
	    fi; \
	  else \
	    echo "....*** $$exe is missing ***"; \
	    echo 0 > $(CHECK_EXAMPLES_STATUS); \
	  fi; \
	done; \
	echo "...Successfully processed $$ok of $$n files."


clean:
	@rm -f *.o *.out *.mout *.diffs

cleanall: clean
	@rm -f $(BINARIES) *.dot *.pdf; \
	if [ "$(RES_DIR)" != "" ]; then \
	  rm -rf $(RES_DIR)/*; \
	fi
