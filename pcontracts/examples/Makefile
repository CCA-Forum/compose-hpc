CXX=g++
FLAGS=-g -O3
INCLUDES=-I.

LDFLAGS=$(FLAGS)
CXXFLAGS=$(INCLUDES) $(FLAGS) -Wno-deprecated

KC_EXE=knapsack
KCPP_EXE=Knapsack

BINARIES=$(KC_EXE) $(KCPP_EXE)

all: $(BINARIES)

check:
	@make checkKnapsacks KT=-1; \
	make checkKnapsacks KT=0; \
	make checkKnapsacks KT=10; \
	make checkKnapsacks KT=; \
	echo "\nDONE"

$(KC_EXE): $(KC_EXE).c
	$(CXX) -c $(CXXFLAGS) -o $(KC_EXE).o $(KC_EXE).c
	$(CXX) -o $(KC_EXE) $(LDFLAGS) $(KC_EXE).o

$(KCPP_EXE): $(KCPP_EXE).cpp
	$(CXX) -c $(CXXFLAGS) -o $(KCPP_EXE).o $(KCPP_EXE).cpp
	$(CXX) -o $(KCPP_EXE) $(LDFLAGS) $(KCPP_EXE).o


checkKnapsacks: $(KC_EXE) $(KCPP_EXE)
	@echo "Running knapsack solvers with target=$$KT..."; \
	./Knapsack $$KT > runKcpp$$KT.out; \
	./knapsack $$KT > runKc$$KT.out; \
	diff runKcpp$$KT.out runKc$$KT.out > runK$$KT.diffs; \
        if [ -s runK$$KT.diffs ]; then \
	  echo "...Results are different"; \
	else \
	  echo "...Results are identical"; \
	fi

clean:
	@rm -f *.o *.out *.diffs

clean-all: clean
	@rm -f $(BINARIES)
