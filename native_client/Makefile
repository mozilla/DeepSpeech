###
### From topdir, first use multistrap to prepare a raspbian jessie armhf root
### $ sudo multistrap -d multistrap-raspbian-jessie -f native_client/multistrap.conf
###
### You can make a tarball after:
### $ touch multistrap-raspbian-jessie.tar && sudo tar cf multistrap-raspbian-jessie.tar multistrap-raspbian-jessie/ && xz multistrap-raspbian-jessie.tar
###
### Then cross-build:
### $ make -C native_client/ TARGET=rpi3 TFDIR=../../tensorflow/tensorflow/
###

.PHONY: clean run bindings

include definitions.mk

default: deepspeech

clean: bindings-clean
	rm -f deepspeech

deepspeech: client.cc
	$(CXX) -o deepspeech $(CFLAGS) $(SOX_CFLAGS) client.cc $(LDFLAGS) $(SOX_LDFLAGS)

bindings-clean:
	rm -rf dist temp_build deepspeech.egg-info MANIFEST.in temp_lib
	rm -f python/model_wrap.cpp python/utils_wrap.cpp python/model.py python/utils.py

bindings-build:
	pip install --quiet $(PYTHON_PACKAGES) wheel
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS)" LDFLAGS="$(RPATH_PYTHON) $(LDFLAGS)" LIBS="$(LIBS)" $(NUMPY_INCLUDE) python ./setup.py build_ext $(PYTHON_PLATFORM_NAME)

MANIFEST.in:
	> $@
	$(call copy_missing_libs,temp_build/*.so,temp_build/lib,$@)

bindings-package: MANIFEST.in
	cat MANIFEST.in
	$(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)

bindings: bindings-build bindings-package

run: deepspeech
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} ./deepspeech ${ARGS}

debug: deepspeech
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} gdb --args ./deepspeech ${ARGS}

install: deepspeech
	install -d ${PREFIX}/lib
	install -m 0644 ${TFDIR}/bazel-bin/tensorflow/libtensorflow_cc.so ${PREFIX}/lib/
	install -m 0644 ${TFDIR}/bazel-bin/native_client/libdeepspeech.so ${PREFIX}/lib/
	install -m 0644 ${TFDIR}/bazel-bin/native_client/libdeepspeech_utils.so ${PREFIX}/lib/
	install -d ${PREFIX}/bin
	install -m 0755 deepspeech ${PREFIX}/bin/

uninstall:
	rm -f ${PREFIX}/bin/deepspeech
	rmdir --ignore-fail-on-non-empty ${PREFIX}/bin
	rm -f ${PREFIX}/lib/libdeepspeech_utils.so
	rm -f ${PREFIX}/lib/libdeepspeech.so
	rm -f ${PREFIX}/lib/libtensorflow_cc.so
	rmdir --ignore-fail-on-non-empty ${PREFIX}/lib
