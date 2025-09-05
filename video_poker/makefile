TARGET = a.out
LIBS = -lm
CC = g++
CFLAGS = -g -Wall
BINDIR = bin

.PHONY: default all clean test

default: $(TARGET)
all: default

APP_SOURCES = main.cc poker.cc neural.cc activations.cc agent.cc
APP_OBJECTS = $(patsubst %.cc, $(BINDIR)/%.o, $(APP_SOURCES))

HEADERS = poker.h neural.h activations.h agent.h

$(BINDIR)/%.o: %.cc $(HEADERS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(APP_OBJECTS)
	$(CC) $(APP_OBJECTS) -Wall $(LIBS) -o $(BINDIR)/$@

POKER_TEST_RUNNER = $(BINDIR)/poker_test_runner
NEURAL_TEST_RUNNER = $(BINDIR)/neural_test_runner

test: test_poker test_neural

test_poker:
	$(CC) $(CFLAGS) -o $(POKER_TEST_RUNNER) poker.cc poker_test.cc
	$(POKER_TEST_RUNNER)

test_neural:
	$(CC) $(CFLAGS) -o $(NEURAL_TEST_RUNNER) neural.cc neural_test.cc activations.cc
	$(NEURAL_TEST_RUNNER)

clean:
	-rm  $(BINDIR)/*.o
	-rm  $(BINDIR)/$(TARGET)
	-rm  $(BINDIR)/poker_test_runner
	-rm  $(BINDIR)/neural_test_runner
