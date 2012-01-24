# things autoconf will fill in
rose_install  = @ROSEDIR@
boost_install = @BOOSTDIR@

# set up compiler and linker flags
CFLAGS += -I$(ROSEDIR)/include
LDFLAGS += -L$(ROSEDIR)/lib

CFLAGS += -I$(BOOSTDIR)/include
LDFLAGS += -L$(BOOSTDIR)/lib

bin_PROGRAMS = test_KVAnnotation test_SXAnnotation test_PaulDecorate \
	example_traversal
example_traversal_SOURCES = example_traversal.cpp
test_KVAnnotation_SOURCES = test_KVAnnotation.cpp
test_PaulDecorate_SOURCES = test_PaulDecorate.cpp
test_SXAnnotation_SOURCES = test_SXAnnotation.cpp
SOURCES=example_traversal.cpp test_KVAnnotation.cpp test_SXAnnotation.cpp \
	test_PaulDecorate.cpp
OBJECTS = $(patsubst %.cpp,%.o,${SOURCES})
EXTRA_DIST=example.paulconf inputs

LIBS=-lrose
LIBS+=-lpaul 
LIBS+=-lfl

LDFLAGS += -L../src
CFLAGS += -I../src

all: $(bin_programs)

.PHONY: $(bin_programs)

example_traversal: example_traversal.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

test_SXAnnotation: test_SXAnnotation.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

test_PaulDecorate: test_PaulDecorate.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

test_KVAnnotation: test_KVAnnotation.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

%.o: %.cpp
	${CXX} ${CFLAGS} -c $<

clean:
	rm -f *.o
	rm -f *~
	rm -f ${bin_programs}
