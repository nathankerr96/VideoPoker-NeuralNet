TARGET = a.out
LIBS = -lm
CC = g++
CFLAGS = -g -Wall
BINDIR = bin

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cc, $(BINDIR)/%.o, $(wildcard *.cc))
HEADERS = $(wildcard *.h)

$(BINDIR)/%.o: %.cc $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $(BINDIR)/$@

clean:
	-rm -f *.o
	-rm -f $(TARGET)