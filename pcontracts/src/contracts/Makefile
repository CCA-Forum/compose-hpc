ROOT_DIR=../..
include $(ROOT_DIR)/src/make_rose.inc


FLAGS=$(BASE_FLAGS)
INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR)

LDFLAGS=$(FLAGS)
CXXFLAGS=$(FLAGS) $(INCLUDES) $(ROSE_CXXFLAGS) $(BASE_CXXFLAGS) 


ROSE_HEADERS = \
	$(INCLUDE_DIR)/RoseHelpers.hpp
ROSE_SOURCES = \
	RoseHelpers.cpp
ROSE_OBJECTS=$(ROSE_SOURCES:.cpp=.o)

CLAUSE_HEADERS = \
	$(ROSE_HEADERS) \
	$(INCLUDE_DIR)/contractOptions.h \
	$(INCLUDE_DIR)/contractClauseTypes.hpp \
	$(INCLUDE_DIR)/ContractsProcessor.hpp
CLAUSE_SOURCES = \
	$(ROSE_SOURCES) \
	ContractsProcessor.cpp
CLAUSE_OBJECTS=$(CLAUSE_SOURCES:.cpp=.o)


LIB=$(LIB_DIR)/$(RUNTIME_LIB)

all: $(CLAUSE_OBJECTS) $(CLAUSE_HEADERS)
	@test -d $(LIB_DIR) || mkdir $(LIB_DIR); \
        ar cr $(LIB) $(CLAUSE_OBJECTS)


# WARNING:  The following assumes unnamed C++ targets are associated
#  with ROSE objects.

.cpp.o: 
	$(CXX) -c $(CXXFLAGS) -o $@ $<

check: 
	@echo "Tests are located in a different directory"

checkall: check

clean:
	rm -f *.o 

cleanall: clean
	rm -f $(LIB)
