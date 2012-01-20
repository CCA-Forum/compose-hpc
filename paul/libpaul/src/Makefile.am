##
##
##
# things autoconf will fill in
rose_install  = @ROSEDIR@
boost_install = @BOOSTDIR@

##
##
##
# set up compiler and linker flags
CFLAGS += -I$(ROSEDIR)/include
LDFLAGS += -L$(ROSEDIR)/lib

CFLAGS += -I$(BOOSTDIR)/include
LDFLAGS += -L$(BOOSTDIR)/lib

##
##
##
# 
lib_LIBRARIES = libpaul.a
libpaul_a_SOURCES = \
        KVAnnotationValue.cpp ssexpr.cpp SXAnnotationValue.cpp Dynamic.cpp \
        PaulDecorate.cpp PaulConfReader.cpp \
        Annotation.h KVAnnotationValue.h ssexpr.h SXAnnotationValue.h \
	PaulConfReader.h PaulDecorate.h Dynamic.h
EXTRA_DIST=scanner.l parser.y lempar.c lemon.c
LIBPAUL=$(lib_LIBRARIES)
SOURCES=$(libpaul_a_SOURCES)
OBJECTS=$(patsubst %.cpp,%.o,${SOURCES})

LIBS=-lrose
LIBS+=-lfl          # Flex!

CXX=g++

all: $(LIBPAUL)

.PHONY: $(LIBPAUL)

$(LIBPAUL): scanner.o parser.o ${OBJECTS}
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

%.o: %.cpp ${HFILES}
	${CXX} ${CFLAGS} -c $<

# Build parser & scanner

CC        = g++

scanner.o: scanner.c scanner.h parser.h
	$(CC) -c $(CFLAGS) scanner.c -o $@

scanner.c scanner.h: scanner.l
	flex $<

parser.c parser.h : parser.y lemon
	./lemon -q $<

parser.o: parser.c parser.h
	$(CC) -c $(CFLAGS) parser.c -o $@

lemon: lemon.c
	gcc lemon.c -o lemon

clean:
	rm -f *.o
	rm -f *~
	rm -f lemon
	rm -f parser.c parser.h scanner.c scanner.h
