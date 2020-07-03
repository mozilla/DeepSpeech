.PHONY: bindings clean workspace_status.cc

include ../definitions.mk

# ARM64 can't find the proper libm.so without this
ifeq ($(TARGET),rpi3-armv8)
LDFLAGS_NEEDED += $(RASPBIAN)/lib/aarch64-linux-gnu/libm.so.6
endif

ifeq ($(OS),Darwin)
GENERATE_DEBUG_SYMS := dsymutil temp_build/temp_build/ds_ctcdecoder/_swigwrapper.*.so
else
GENERATE_DEBUG_SYMS :=
endif

ifeq ($(findstring _NT,$(OS)),_NT)
	ARCHIVE_EXT := lib
else
	ARCHIVE_EXT := a
endif


all: bindings

clean:
	rm -rf dist temp_build ds_ctcdecoder.egg-info
	rm -f swigwrapper_wrap.cpp swigwrapper.py

# Enforce PATH here because swig calls from build_ext looses track of some
# variables over several runs
bindings: clean ds-swig
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.33.6 setuptools==39.1.0
	DISTUTILS_USE_SDK=1 PATH=$(DS_SWIG_BIN_PATH):$(TOOLCHAIN):$$PATH SWIG_LIB="$(SWIG_LIB)" AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" TFDIR="$(TFDIR)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build_ext $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	find temp_build -type f -name "*.o" -delete
	DISTUTILS_USE_SDK=1 AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" TFDIR="$(TFDIR)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	rm -rf temp_build

bindings-debug: clean ds-swig
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.33.6 setuptools==39.1.0
	DISTUTILS_USE_SDK=1 PATH=$(DS_SWIG_BIN_PATH):$(TOOLCHAIN):$$PATH SWIG_LIB="$(SWIG_LIB)" AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS) -DDEBUG" LDFLAGS="$(LDFLAGS_NEEDED)" TFDIR="$(TFDIR)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build_ext --debug $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	$(GENERATE_DEBUG_SYMS)
	find temp_build -type f -name "*.o" -delete
	DISTUTILS_USE_SDK=1 AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) LIBEXE=$(LIBEXE) CFLAGS="$(CFLAGS) $(CXXFLAGS) -DDEBUG" LDFLAGS="$(LDFLAGS_NEEDED)" TFDIR="$(TFDIR)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	rm -rf temp_build
