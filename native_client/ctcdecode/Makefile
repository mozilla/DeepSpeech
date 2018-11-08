.PHONY: bindings clean

include ../definitions.mk

NUM_PROCESSES ?= 1

all: bindings

clean:
	rm -rf dist temp_build ds_ctcdecoder.egg-info
	rm -f swigwrapper_wrap.cpp swigwrapper.py

bindings: clean
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.31.0 setuptools==39.1.0
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build --num_processes $(NUM_PROCESSES) $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	find temp_build -type f -name "*.o" -delete
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS) $(CXXFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel --num_processes $(NUM_PROCESSES) $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)
	rm -rf temp_build
