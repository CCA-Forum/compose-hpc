# Note: If the ROSE header and library files are not in standard places,
# define ROSEINSTALL in your environment:

ifdef SEXPRINSTALL
CFLAGS += -I$(SEXPRINSTALL)
LDFLAGS += -L$(SEXPRINSTALL)
endif
ifdef ROSEINSTALL
CFLAGS += -I$(ROSEINSTALL)/include
LDFLAGS += -L$(ROSEINSTALL)/lib
endif
ifdef BOOSTINSTALL
CFLAGS += -I$(BOOSTINSTALL)/include
LDFLAGS += -L$(BOOSTINSTALL)/lib
endif

EXEC=test_PaulDecorate
CCHELPERS=KVAnnotationValue.cpp SXAnnotationValue.cpp Dynamic.cpp \
          PaulDecorate.cpp
CCFILES=${EXEC}.cpp ${CCHELPERS}

HFILES=$(patsubst %.cpp,%.h,${CCHELPERS})
OFILES=$(patsubst %.cpp,%.o,${CCFILES})
LIBS=-lrose
LIBS+=-lfl          # Flex!
LIBS+=-lsexp        # sexpr lib
CXX=g++
SOURCES=${CCFILES}
OUTS=$(addprefix out/, $(basename $(notdir ${SOURCES})))
OUTS_O=$(addprefix out-o/, $(basename $(notdir ${SOURCES})))

# CFLAGS += -Wall -O2 -ansi

.PHONY: main
main: ${EXEC}

${EXEC}: ${OFILES} scanner.o parser.o
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

%.o: %.cc ${HFILES}
	${CXX} ${CFLAGS} -c $<

%.o: %.cpp ${HFILES}
	${CXX} ${CFLAGS} -c $<

run: out

.PHONY: out
out: ${EXEC}
	rm -rf $@
	mkdir $@
	for f in ${EXAMPLES} ${TESTS} ; do b=$$(basename "$$f") ; echo ----- $$b ----- ; ./${EXEC} -rose:o out/$$b $$f ; done

out/%: out/%.c
	mpicc -o $@ $<

.PHONY: mpibuild
mpibuild: ${OUTS}

# Build parser & scanner

CC        = g++

scanner.o: scanner.c scanner.h parser.h
	$(CC) -c $(CFLAGS) scanner.c -o $@

scanner.c scanner.h: scanner.l
	flex $<

parser.c parser.h : parser.y lemon
	./lemon -q $<

lemon: lemon.c
	gcc lemon.c -o lemon

# Create skeletons via outlining, not removing

.PHONY: out-o
out-o: ${EXEC}
	rm -rf $@
	mkdir $@
	for f in ${EXAMPLES} ${TESTS} ; do b=$$(basename "$$f") ; echo ----- $$b ----- ; ./${EXEC} -rose:o out-o/$$b -local:outline $$f ; done

out-o/%: out-o/%.c
	mpicc -o $@ $<

.PHONY: mpibuild
mpibuild-o: ${OUTS_O}

  # ^ not fully successful, use "make -k mpibuild-o" for now.

.PHONY: compare
compare: ${OUTS}
	@for f in out/*.c ; do b=$$(basename "$$f") ; if [ -e expected/$$b.expected ] ; then diff $$f expected/$$b.expected ; fi ; done

.PHONY: test
test: ${EXEC}
	${MAKE} run
	${MAKE} mpibuild
	${MAKE} compare

TAGS: ${CCFILES} ${HFILES}
	etags $^

show:
	@echo $($(VALUE))              # note: this expands!
