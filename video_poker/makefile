TARGET = a.out
LIBS = -lm
CC = g++
CFLAGS = -g -Wall -std=c++20 -O2 -I. -x c++
BINDIR = bin

.PHONY: default all clean test lint

default: $(TARGET)
all: default

# --- Source File Definitions ---
ALL_SOURCES = $(shell find . -name '*.cc')
HEADERS = $(shell find . -name '*.h')

# --- Application Build ---
# Filter out test files from all sources
APP_SOURCES = $(filter-out $(shell find . -name '*_test.cc') ./main_test.cc, $(ALL_SOURCES))
APP_OBJECTS = $(patsubst %.cc, $(BINDIR)/%.o, $(APP_SOURCES))

# --- Test Build ---
TEST_TARGET = $(BINDIR)/test.out
# Get all _test.cc files
TEST_FILES = $(filter $(shell find . -name '*_test.cc'), $(ALL_SOURCES))
# App sources needed for tests (all app .cc files except main.cc)
TEST_APP_SOURCES = $(filter-out ./main.cc, $(APP_SOURCES))
# All object files needed for the test executable
TEST_OBJECTS = $(patsubst %.cc, $(BINDIR)/%.o, ./main_test.cc $(TEST_FILES) $(TEST_APP_SOURCES))

# --- Generic Build Rules ---
$(BINDIR)/%.o: %.cc $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# --- Target-specific Rules ---
$(TARGET): $(APP_OBJECTS)
	$(CC) $^ -Wall $(LIBS) -o $(BINDIR)/$@

test: $(TEST_TARGET)
	@echo "Running tests..."
	@$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJECTS)
	$(CC) $^ -Wall $(LIBS) -o $@

# --- Maintenance Rules ---
LINT_SOURCES = $(ALL_SOURCES)

lint:
	@echo "Running clang-tidy on all source files..."
	@for file in $(LINT_SOURCES); do \
		echo "  Linting $$file..."; \
		clang-tidy $$file -- $(CFLAGS); \
	done

clean:
	-rm -f $(BINDIR)/*.o
	-rm -f $(BINDIR)/$(TARGET)
	-rm -f $(TEST_TARGET)
