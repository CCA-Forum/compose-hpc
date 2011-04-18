SOURCES    = scanner.l parser.y main.cpp \
						 Annotation.cpp \
						 Dynamic.cpp \
						 CommentVisitor.cpp
BIN        = paul

OBJECTS   = scanner.o parser.o Annotation.o Dynamic.o CommentVisitor.o main.o
CC        = g++
CFLAGS    = -I$(ROSE_HOME)/include -I$(BOOST_HOME)/include
LDFLAGS   = -L$(ROSE_HOME)/lib -L$(BOOST_HOME)/lib -lm -lfl -lrose

all: lemon $(SOURCES) $(BIN)

$(BIN): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

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
	rm -rf $(BIN) $(OBJECTS) *.ti parser.c parser.h scanner.c scanner.h lemon
