SOURCES     = Main.cpp TestVisitor.cpp
EXECUTABLE  = comments

OBJECTS=$(SOURCES:.cpp=.o)

CC        = g++
CFLAGS    = 
LDFLAGS   = -L$(ROSE_HOME)/lib -L$(BOOST_HOME)/lib -lrose

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) -c $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

lemon: lemon.c
  $(CC) lemon.c -o lemon

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) lemon
