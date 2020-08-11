.PHONY: bindings clean workspace_status.cc

include ../definitions.mk

NUM_PROCESSES ?= 1

# ARM64 can't find the proper libm.so without this
ifeq ($(TARGET),rpi3-armv8)
LDFLAGS_NEEDED += $(RASPBIAN)/lib/aarch64-linux-gnu/libm.so.6
endif

ifeq ($(OS),Darwin)
GENERATE_DEBUG_SYMS := dsymutil temp_build/temp_build/mvs_ctcdecoder/_swigwrapper.*.so
else
GENERATE_DEBUG_SYMS :=
endif

ifeq ($(findstring _NT,$(OS)),_NT)
	ARCHIVE_EXT := lib
else
	ARCHIVE_EXT := a
endif

FIRST_PARTY := first_party.$(ARCHIVE_EXT)
THIRD_PARTY := third_party.$(ARCHIVE_EXT)


all: bindings

clean-keep-third-party:
	rm -rf dist temp_build mvs_ctcdecoder.egg-info
	rm -f swigwrapper_wrap.cpp swigwrapper.py $(FIRST_PARTY)

clean: clean-keep-third-party
	rm -f $(THIRD_PARTY)
	rm workspace_status.cc
	rm -fr bazel-out/

workspace_status.cc:
	mkdir -p bazel-out/ && \
		../bazel_workspace_status_cmd.sh > bazel-out/stable-status.txt && \
		../gen_workspace_status.sh > $@

# Enforce PATH here because swig calls from build_ext looses track of some
# variables over several runs
bindings: clean-keep-third-party workspace_status.cc ds-swig
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.33.6 setuptools==39.1.0
	DISTUTILS_USE_SDK=1 PATH=$(DS_SWIG_BIN_PATH):$(TOOLCHAIN):$$PATH SWIG_LIB="$(SWIG_LIB)" AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build_ext --num_processes $(NUM_PROCESSES) $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	find temp_build -type f -name "*.o" -delete
	DISTUTILS_USE_SDK=1 AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	rm -rf temp_build

bindings-debug: clean-keep-third-party workspace_status.cc ds-swig
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.33.6 setuptools==39.1.0
	DISTUTILS_USE_SDK=1 PATH=$(DS_SWIG_BIN_PATH):$(TOOLCHAIN):$$PATH SWIG_LIB="$(SWIG_LIB)" AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS) -DDEBUG" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build_ext --debug --num_processes $(NUM_PROCESSES) $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	$(GENERATE_DEBUG_SYMS)
	find temp_build -type f -name "*.o" -delete
	DISTUTILS_USE_SDK=1 AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS) -DDEBUG" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	rm -rf temp_build
