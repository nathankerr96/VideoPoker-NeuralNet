TARGET = a.out
LIBS = -lm
CC = g++
CFLAGS = -g -Wall
BINDIR = bin

.PHONY: default all clean test

default: $(TARGET)
all: default

APP_SOURCES = main.cc poker.cc
APP_OBJECTS = $(patsubst %.cc, $(BINDIR)/%.o, $(APP_SOURCES))

HEADERS = poker.h

$(BINDIR)/%.o: %.cc $(HEADERS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(APP_OBJECTS)
	$(CC) $(APP_OBJECTS) -Wall $(LIBS) -o $(BINDIR)/$@

test:
	$(CC) $(CFLAGS) -o $(BINDIR)/test_runner poker.cc poker_test.cc
	$(BINDIR)/test_runner

clean:
	-rm -f $(BINDIR)/*.o
	-rm -f $(BINDIR)/$(TARGET)
	-rm -f $(BINDIR)/test_runner
