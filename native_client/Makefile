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
ifeq ($(OS),Darwin)
	install_name_tool -change $$TASKCLUSTER_TASK_DIR/homebrew/opt/sox/lib/libsox.3.dylib @rpath/libsox.3.dylib deepspeech
	install_name_tool -change bazel-out/local-opt/bin/native_client/libdeepspeech.so @rpath/libdeepspeech.so deepspeech
	install_name_tool -change bazel-out/local-opt/bin/native_client/libdeepspeech_utils.so @rpath/libdeepspeech_utils.so deepspeech
endif

bindings-clean:
	rm -rf dist temp_build deepspeech.egg-info MANIFEST.in temp_lib
	rm -f python/model_wrap.cpp python/utils_wrap.cpp python/model.py python/utils.py

bindings-build:
	pip install --quiet $(PYTHON_PACKAGES) wheel
	AS=$(AS) CC=$(CC) CXX=$(CXX) LD=$(LD) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS_NEEDED) $(RPATH_PYTHON)" MODEL_LDFLAGS="$(LDFLAGS_DIRS)" UTILS_LDFLAGS="-L${TFDIR}/bazel-bin/native_client" MODEL_LIBS="$(LIBS)" $(NUMPY_INCLUDE) python ./setup.py build_ext $(PYTHON_PLATFORM_NAME)

MANIFEST.in: bindings-build
	> $@
	$(call copy_missing_libs,temp_build/deepspeech/*.so,temp_build/deepspeech/lib,$@)

bindings-package: MANIFEST.in
	cat MANIFEST.in
	rm temp_build/python/*_wrap.o
	$(NUMPY_INCLUDE) python ./setup.py bdist_wheel $(PYTHON_PLATFORM_NAME) $(SETUP_FLAGS)

bindings: bindings-build bindings-package

run: deepspeech
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} ./deepspeech ${ARGS}

debug: deepspeech
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} gdb --args ./deepspeech ${ARGS}

install: deepspeech
	install -d ${PREFIX}/lib
	install -m 0644 ${TFDIR}/bazel-bin/tensorflow/libtensorflow_cc.so ${PREFIX}/lib/
	install -m 0644 ${TFDIR}/bazel-bin/tensorflow/libtensorflow_framework.so ${PREFIX}/lib/
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
	rm -f ${PREFIX}/lib/libtensorflow_framework.so
	rmdir --ignore-fail-on-non-empty ${PREFIX}/lib
