TARGET = a.out
LIBS = -lm
CC = g++
CFLAGS = -g -Wall -std=c++20 -O2 -I. -x c++
BINDIR = bin

.PHONY: default all clean test lint

default: $(TARGET)
all: default

APP_SOURCES = $(filter-out $(shell find . -name '*_test.cc'), $(shell find . -name '*.cc'))
APP_OBJECTS = $(patsubst %.cc, $(BINDIR)/%.o, $(APP_SOURCES))

HEADERS = $(shell find . -name '*.h')

$(BINDIR)/%.o: %.cc $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(APP_OBJECTS)
	$(CC) $(APP_OBJECTS) -Wall $(LIBS) -o $(BINDIR)/$@

POKER_TEST_RUNNER = $(BINDIR)/poker_test_runner

test: test_poker

test_poker:
	$(CC) $(CFLAGS) -o $(POKER_TEST_RUNNER) poker.cc poker_test.cc
	$(POKER_TEST_RUNNER)

LINT_SOURCES = $(shell find . -name '*.cc')

lint:
	@echo "Running clang-tidy on all source files..."
	@for file in $(LINT_SOURCES); do \
		echo "  Linting $$file..."; \
		clang-tidy $$file -- $(CFLAGS); \
	done

clean:
	-rm  $(BINDIR)/*.o
	-rm  $(BINDIR)/$(TARGET)
	-rm  $(BINDIR)/poker_test_runner
