#ben settings:
# spatch is in /home/baallan/coke on ubuntu 1010220 vm
BOOST_HOME=/home/baallan/diversity/roses/boost1450
ROSE_HOME=/home/baallan/diversity/roses/b1210
#end ben
SOURCES    = scanner.l parser.y main.cpp \
						 Annotation.cpp \
						 Dynamic.cpp \
						 CommentVisitor.cpp \
						 Transforms.cpp
BIN       = paul
ST        = blas2cublas/SimpleTranslator

B2CBSRC   := $(wildcard blas2cublas/src/*.cpp)
B2CBOBJ   := $(addprefix , $(B2CBSRC:.cpp=.o))

B2CBLIB   = blas2cublas/src/libb2cb.a
OBJECTS   = scanner.o parser.o Annotation.o Dynamic.o CommentVisitor.o main.o Transforms.o
CC        = g++
CFLAGS    = -I$(ROSE_HOME)/include -I$(BOOST_HOME)/include 
LDFLAGS   = -L$(ROSE_HOME)/lib -L$(BOOST_HOME)/lib -lm -lfl -lrose 

all: lemon $(SOURCES) $(ST) $(B2CBLIB) $(BIN) 

$(BIN): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) blas2cublas/src/libb2cb.a

$(ST):  blas2cublas/SimpleTranslator.C
	$(CC) $(CFLAGS) blas2cublas/SimpleTranslator.C -o $@ $(LDFLAGS)

$(B2CBLIB): $(B2CBOBJ) $(B2CBSRC:%.cpp=%.o)
	rm -f $@
	ar -cvq $@ $(B2CBSRC:%.cpp=%.o)

blas2cublas/src/%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

scanner.o: scanner.c scanner.h parser.h
	$(CC) -c $(CFLAGS) scanner.c -o $@

scanner.c scanner.h: scanner.l
	flex $<

parser.c parser.h : parser.y
	./lemon -q $<

lemon: lemon.c
	gcc lemon.c -o lemon

clean:
	rm -rf $(BIN) $(OBJECTS) $(ST) lemon parser.c parser.h scanner.c scanner.h \
	       *.ti *.dot *~ 

cleanb2cb:
	rm -rf blas2cublas/src/*.o
