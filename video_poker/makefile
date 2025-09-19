TARGET = a.out
LIBS = -lm
CC = g++
CFLAGS = -g -Wall -std=c++20 -O2
BINDIR = bin

.PHONY: default all clean test

default: $(TARGET)
all: default

APP_SOURCES = main.cc poker.cc neural.cc activations.cc agent.cc decision.cc baseline.cc trainer.cc optimizer.cc
APP_OBJECTS = $(patsubst %.cc, $(BINDIR)/%.o, $(APP_SOURCES))

HEADERS = poker.h neural.h activations.h agent.h decision.h baseline.h trainer.h hyperparams.h optimizer.h

$(BINDIR)/%.o: %.cc $(HEADERS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(APP_OBJECTS)
	$(CC) $(APP_OBJECTS) -Wall $(LIBS) -o $(BINDIR)/$@

POKER_TEST_RUNNER = $(BINDIR)/poker_test_runner

test: test_poker

test_poker:
	$(CC) $(CFLAGS) -o $(POKER_TEST_RUNNER) poker.cc poker_test.cc
	$(POKER_TEST_RUNNER)

clean:
	-rm  $(BINDIR)/*.o
	-rm  $(BINDIR)/$(TARGET)
	-rm  $(BINDIR)/poker_test_runner
