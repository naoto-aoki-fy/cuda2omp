PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
INCLUDEDIR ?= $(PREFIX)/include
PYTHON ?= python3

.PHONY: all test check install

all:
	@echo "cuda2omp is ready (no build step required)"

test check:
	$(PYTHON) tests/test_cuda2omp.py

install:
	install -d "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(INCLUDEDIR)"
	install -m 755 cuda2omp "$(DESTDIR)$(BINDIR)/cuda2omp"
	install -m 644 runtime/cuda2omp_runtime.hpp \
		"$(DESTDIR)$(INCLUDEDIR)/cuda2omp_runtime.hpp"
