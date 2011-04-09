SOURCES    = Grammar.c Main.c
EXECUTABLE = calc

OBJECTS   = $(SOURCES:.c=.o)
CC        = gcc
CFLAGS    = 
LDFLAGS   =

all: lemon $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.y.c:
	./lemon $<

lemon: lemon.c
	$(CC) lemon.c -o lemon

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) Grammar.c Grammar.h lemon
