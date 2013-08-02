ROOT_DIR=../..
include $(ROOT_DIR)/make_rose.inc
include $(ROOT_DIR)/make_test.inc


FLAGS=$(BASE_FLAGS)
INCLUDES=$(BASE_INCLUDES) -I$(RT_DIR)

LDFLAGS=$(FLAGS)
CXXFLAGS=$(FLAGS) $(INCLUDES) $(ROSE_CXXFLAGS) $(BASE_CXXFLAGS) 


ROSE_SOURCES = \
	RoseHelpers.cpp
ROSE_OBJECTS=$(ROSE_SOURCES:.cpp=.o)
CLAUSE_SOURCES = \
	$(ROSE_SOURCES) \
	ContractsProcessor.cpp
CLAUSE_OBJECTS=$(CLAUSE_SOURCES:.cpp=.o)


LIB=$(LIB_DIR)/$(RUNTIME_LIB)

all: $(CLAUSE_OBJECTS)
	@test -d $(LIB_DIR) || mkdir $(LIB_DIR); \
        ar cr $(LIB) $(CLAUSE_OBJECTS)


# WARNING:  The following assumes unnamed C++ targets are associated
#  with ROSE objects.

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

check: 

checkall: 

clean:
	rm -f *.o 

cleanall: clean
	rm -f $(LIB)
