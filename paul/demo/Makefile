SOURCES    = scanner.l parser.y main.cpp \
						 Annotation.cpp \
						 Dynamic.cpp \
						 CommentVisitor.cpp \
						 Transforms.cpp
BIN       = paul
ST        = SimpleTranslator

B2CBSRC   := $(wildcard blas2cublas/src/*.cpp)
B2CBOBJ   := $(addprefix , $(B2CBSRC:.cpp=.o))

TASCELSRC   := $(wildcard tascel/src/*.cpp)
TASCELOBJ   := $(addprefix , $(TASCELSRC:.cpp=.o))

B2CBLIB   = blas2cublas/src/libb2cb.a
TASCELLIB = tascel/src/libtascel.a
OBJECTS   = scanner.o parser.o Annotation.o Dynamic.o CommentVisitor.o main.o Transforms.o
CC        = g++
CFLAGS    = -I$(ROSE_HOME)/include -I$(BOOST_HOME)/include 
LDFLAGS   = -L$(ROSE_HOME)/lib -L$(BOOST_HOME)/lib -lm -lfl -lrose 

all: lemon $(SOURCES) $(ST) $(B2CBLIB) $(TASCELLIB) $(BIN) 

$(BIN): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) blas2cublas/src/libb2cb.a tascel/src/libtascel.a

$(ST):  SimpleTranslator.cpp
	$(CC) $(CFLAGS) SimpleTranslator.cpp -o $@ $(LDFLAGS)

$(B2CBLIB): $(B2CBOBJ) $(B2CBSRC:%.cpp=%.o)
	rm -f $@
	ar -cvq $@ $(B2CBSRC:%.cpp=%.o)
	
$(TASCELLIB): $(TASCELOBJ) $(TASCELSRC:%.cpp=%.o)
	rm -f $@
	ar -cvq $@ $(TASCELSRC:%.cpp=%.o)

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
	rm -rf blas2cublas/src/*.o blas2cublas/src/libb2cb.a

cleanTascel:
	rm -rf tascel/src/*.o tascel/src/libtascel.a
	
cleanAll: clean cleanb2cb cleanTascel