.PHONY: bindings clean workspace_status.cc

include ../definitions.mk

NUM_PROCESSES ?= 1

# ARM64 can't find the proper libm.so without this
ifeq ($(TARGET),rpi3-armv8)
LDFLAGS_NEEDED += $(RASPBIAN)/lib/aarch64-linux-gnu/libm.so.6
endif

ifeq ($(OS),Darwin)
GENERATE_DEBUG_SYMS := dsymutil temp_build/temp_build/ds_ctcdecoder/_swigwrapper.*.so
else
GENERATE_DEBUG_SYMS :=
endif

all: bindings

clean-keep-third-party:
	rm -rf dist temp_build ds_ctcdecoder.egg-info
	rm -f swigwrapper_wrap.cpp swigwrapper.py first_party.a

clean: clean-keep-third-party
	rm -f third_party.a
	rm workspace_status.cc
	rm -fr bazel-out/

workspace_status.cc:
	mkdir -p bazel-out/ && \
		../bazel_workspace_status_cmd.sh > bazel-out/stable-status.txt && \
		../gen_workspace_status.sh > $@

bindings: clean-keep-third-party workspace_status.cc
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.33.6 setuptools==39.1.0
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build_ext --num_processes $(NUM_PROCESSES) $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	find temp_build -type f -name "*.o" -delete
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	rm -rf temp_build

bindings-debug: clean-keep-third-party workspace_status.cc
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.33.6 setuptools==39.1.0
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS) $(CXXFLAGS) -DDEBUG" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build_ext --debug --num_processes $(NUM_PROCESSES) $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	$(GENERATE_DEBUG_SYMS)
	find temp_build -type f -name "*.o" -delete
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS) $(CXXFLAGS) -DDEBUG" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	rm -rf temp_build
