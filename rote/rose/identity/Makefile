SOURCES     = identity.cpp
EXECUTABLE  = identity

OBJECTS=$(SOURCES:.cpp=.o)

CC        = g++
CFLAGS    = -c -I$(ROSE_HOME)/include -I$(BOOST_HOME)/include
LDFLAGS   = -L$(ROSE_HOME)/lib -L$(BOOST_HOME)/lib -lrose

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS)
