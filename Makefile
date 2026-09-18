PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
INCLUDEDIR ?= $(PREFIX)/include
LIBDIR ?= $(PREFIX)/lib
RESOURCE_DIR ?= $(LIBDIR)/cuda2omp
PYTHON ?= python3
CXX ?= c++
LLVM_CONFIG ?= llvm-config
BUILD_DIR ?= build
NATIVE_TOOL := $(BUILD_DIR)/cuda2omp-tool
PUBLIC_HEADERS := runtime/cuda2omp_runtime.hpp runtime/cuda_runtime.h
COMPATIBILITY_HEADERS := $(wildcard runtime/*_shim.hpp)
HAVE_LLVM_CONFIG := $(shell command -v "$(LLVM_CONFIG)" >/dev/null 2>&1 && echo yes)

.PHONY: all native test check benchmark installcheck install clean

all: native

# LLVM development packages are deliberately an optional dependency while the
# native frontend is brought to feature parity with the Python implementation.
ifeq ($(HAVE_LLVM_CONFIG),yes)
native: $(NATIVE_TOOL)

$(NATIVE_TOOL): native/cuda2omp_tool.cpp
	mkdir -p "$(BUILD_DIR)"
	$(CXX) $$($(LLVM_CONFIG) --cxxflags) -std=c++17 $< -o $@ \
		$$($(LLVM_CONFIG) --ldflags) -lclang-cpp \
		$$($(LLVM_CONFIG) --libs --system-libs)
else
native:
	@echo "cuda2omp: $(LLVM_CONFIG) not found; skipping optional native frontend"
endif

test check:
	$(PYTHON) tests/test_cuda2omp.py

benchmark:
	$(PYTHON) benchmarks/barrier_free_launch.py

# Kept as a named packaging check so distributors can exercise the installed
# layout independently of the rest of the source-tree test suite.
installcheck:
	$(PYTHON) tests/test_cuda2omp.py Tests.test_staged_install_is_self_contained

install:
	install -d "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(INCLUDEDIR)" \
		"$(DESTDIR)$(RESOURCE_DIR)"
	install -m 755 cuda2omp "$(DESTDIR)$(BINDIR)/cuda2omp"
	install -m 644 $(PUBLIC_HEADERS) "$(DESTDIR)$(INCLUDEDIR)/"
	install -m 644 $(COMPATIBILITY_HEADERS) "$(DESTDIR)$(RESOURCE_DIR)/"
	@if test -x "$(NATIVE_TOOL)"; then \
		install -m 755 "$(NATIVE_TOOL)" "$(DESTDIR)$(BINDIR)/cuda2omp-tool"; \
	fi

clean:
	rm -rf "$(BUILD_DIR)"
