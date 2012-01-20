# things autoconf will fill in
rose_install  = @ROSEDIR@
boost_install = @BOOSTDIR@

# set up compiler and linker flags
CFLAGS += -I$(ROSEDIR)/include
LDFLAGS += -L$(ROSEDIR)/lib

CFLAGS += -I$(BOOSTDIR)/include
LDFLAGS += -L$(BOOSTDIR)/lib

bin_programs = test_KVAnnotation test_SXAnnotation
test_KVAnnotation_SOURCES = test_KVAnnotation.cpp
test_SXAnnotation_SOURCES = test_SXAnnotation.cpp
SOURCES=$(test_KVAnnotation_SOURCES) $(test_SXAnnotation_SOURCES)
OBJECTS = $(patsubst %.cpp,%.o,${SOURCES})

LIBS=-lrose
LIBS+=-lfl
LIBS+=-lpaul

LDFLAGS += -L../src
CFLAGS += -I../src

all: $(bin_programs)

.PHONY: $(bin_programs)

test_SXAnnotation: test_SXAnnotation.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

test_KVAnnotation: test_KVAnnotation.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

%.o: %.cpp
	${CXX} ${CFLAGS} -c $<

clean:
	rm -f *.o
	rm -f *~
	rm -f ${bin_programs}
