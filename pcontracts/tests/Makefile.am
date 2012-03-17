# things autoconf will fill in
rose_install  = @ROSEDIR@
boost_install = @BOOSTDIR@

# set up compiler and linker flags
CFLAGS += -I$(ROSEDIR)/include -I$(BOOSTDIR)/include -I../include
LDFLAGS += -L$(ROSEDIR)/lib -L$(BOOSTDIR)/lib

InvariantAnnotationExe = test_InvariantAnnotation
PaulContractsDecoratorExe = test_PaulContractsDecorator
PreconditionAnnotationExe = test_PreconditionAnnotation
PostconditionAnnotationExe = test_PostconditionAnnotation
SimpleContractsPrinterExe = test_SimpleContractsPrinter

bin_PROGRAMS = \
	$(InvariantAnnotation) \
	$(PaulContractsDecorator) \
	$(PostconditionAnnotation) \
	$(PreconditionAnnotation) \
	$(SimpleContractsPrinter) 

InvariantAnnotationSrc = $(InvariantAnnotation).cpp
PaulContractsDecorateSrc = $(PaulContractsDecorate).cpp
PostconditionAnnotationSrc = $(PostconditionAnnotation).cpp
PreconditionAnnotationSrc = $(PreconditionAnnotation).cpp
SimpleContractsPrinterSrc = $(SimpleContractsPrinter).cpp

InvariantAnnotationObj = $(InvariantAnnotation).o
PaulContractsDecorateObj = $(PaulContractsDecorate).o
PostconditionAnnotationObj = $(PostconditionAnnotation).o
PreconditionAnnotationObj = $(PreconditionAnnotation).o
SimpleContractsPrinterObj = $(SimpleContractsPrinter).o

SOURCES = \
	SimpleContractsPrinter.cpp \
	$(InvariantAnnotationSrc) \
	$(PaulContractsDecoratorSrc) \
	$(PostconditionAnnotationSrc) \
	$(PreconditionAnnotationSrc) \
	$(SimpleContractsPrinterSrc) 

OBJECTS = $(patsubst %.cpp,%.o,${SOURCES})

# ToDo/TBD: Should the paulconf file be kept local
EXTRA_DIST=../config/Contracts.paulconf inputs

LIBS=-lrose -lpaul 

# ToDo/TBD: What is the following library?
LIBS+=-lfl

LDFLAGS += -L../src
CFLAGS += -I../src -I../include

all: $(bin_programs)

.PHONY: $(bin_programs)

%.o: %.cpp
	${CXX} ${CFLAGS} -c $<

$(InvariantAnnotation): $(InvariantAnnotationObj)
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

$(PaulContractsDecorate): $(PaulContractsDecorateObj)
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

$(PostconditionAnnotation): $(PostconditionAnnotationObj)
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

$(PreconditionAnnotation): $(PreconditionAnnotationObj)
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

$(SimpleContractsPrinter): $(SimpleContractsPrinterObj) SimpleContractsPrinter.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

clean:
	rm -f *.o *~ ${bin_programs}
