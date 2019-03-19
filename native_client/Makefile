###
### From topdir, first use multistrap to prepare a raspbian stretch armhf root
### $ sudo multistrap -d multistrap-raspbian-stretch -f native_client/multistrap_raspbian_stretch.conf
###
### You can make a tarball after:
### $ touch multistrap-raspbian-stretch.tar && sudo tar cf multistrap-raspbian-stretch.tar multistrap-raspbian-stretch/ && xz multistrap-raspbian-stretch.tar
###
### Then cross-build:
### $ make -C native_client/ TARGET=rpi3 TFDIR=../../tensorflow/tensorflow/
###

.PHONY: clean run print-toolchain

include definitions.mk

default: $(DEEPSPEECH_BIN)

clean: bindings-clean
	rm -f deepspeech

$(DEEPSPEECH_BIN): client.cc
	$(CXX) $(CFLAGS) $(CFLAGS_DEEPSPEECH) $(SOX_CFLAGS) client.cc $(LDFLAGS) $(SOX_LDFLAGS)
ifeq ($(OS),Darwin)
	install_name_tool -change $$TASKCLUSTER_TASK_DIR/homebrew/opt/sox/lib/libsox.3.dylib @rpath/libsox.3.dylib deepspeech
	install_name_tool -change bazel-out/local-opt/bin/native_client/libdeepspeech.so @rpath/libdeepspeech.so deepspeech
endif

run: $(DEEPSPEECH_BIN)
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} ./deepspeech ${ARGS}

debug: $(DEEPSPEECH_BIN)
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} gdb --args ./deepspeech ${ARGS}

install: $(DEEPSPEECH_BIN)
	install -d ${PREFIX}/lib
	install -m 0644 ${TFDIR}/bazel-bin/native_client/libdeepspeech.so ${PREFIX}/lib/
	install -d ${PREFIX}/bin
	install -m 0755 deepspeech ${PREFIX}/bin/

uninstall:
	rm -f ${PREFIX}/bin/deepspeech
	rmdir --ignore-fail-on-non-empty ${PREFIX}/bin
	rm -f ${PREFIX}/lib/libdeepspeech.so
	rmdir --ignore-fail-on-non-empty ${PREFIX}/lib

print-toolchain:
	@echo $(TOOLCHAIN)
