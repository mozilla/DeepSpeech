.PHONY: bindings

include ../definitions.mk

bindings-clean:
	rm -rf dist temp_build deepspeech.egg-info MANIFEST.in temp_lib
	rm -f impl_wrap.cpp impl.py

bindings-build:
	pip install --quiet $(PYTHON_PACKAGES) wheel==0.31.0 setuptools==39.1.0
	PATH=$(TOOLCHAIN):$$PATH AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED) $(RPATH_PYTHON)" MODEL_LDFLAGS="$(LDFLAGS_DIRS)" MODEL_LIBS="$(LIBS)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py build_ext $(PYTHON_PLATFORM_NAME)

MANIFEST.in: bindings-build
	> $@
	$(call copy_missing_libs,temp_build/deepspeech/*.so,temp_build/deepspeech/lib,$@)

# On Unix, _wrap.o gets generated
# On Windows, _wrap.obj it is
bindings-package: MANIFEST.in
	cat MANIFEST.in
	rm -f temp_build/*_wrap.o temp_build/Release/*_wrap.obj
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED) $(RPATH_PYTHON)" MODEL_LDFLAGS="$(LDFLAGS_DIRS)" MODEL_LIBS="$(LIBS)" $(PYTHON_PATH) $(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)

bindings: bindings-build bindings-package
