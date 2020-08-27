###
### From topdir, first use multistrap to prepare a raspbian buster armhf root
### $ multistrap -d multistrap-raspbian-buster -f native_client/multistrap_raspbian_buster.conf
###
### You can make a tarball after:
### $ touch multistrap-raspbian-buster.tar && sudo tar cf multistrap-raspbian-buster.tar multistrap-raspbian-buster/ && xz multistrap-raspbian-buster.tar
###
### Then cross-build:
### $ make -C native_client/ TARGET=rpi3 TFDIR=../../tensorflow/tensorflow/
###

.PHONY: clean run print-toolchain

include definitions.mk

default: $(DEEPSPEECH_BIN)

clean:
	rm -f deepspeech

$(DEEPSPEECH_BIN): client.cc Makefile
	$(CXX) $(CFLAGS) $(CFLAGS_DEEPSPEECH) $(SOX_CFLAGS) client.cc $(LDFLAGS) $(SOX_LDFLAGS)
ifeq ($(OS),Darwin)
	install_name_tool -change bazel-out/local-opt/bin/native_client/libdeepspeech.so @rpath/libdeepspeech.so deepspeech
endif

run: $(DEEPSPEECH_BIN)
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} ./deepspeech ${ARGS}

debug: $(DEEPSPEECH_BIN)
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} gdb --args ./deepspeech ${ARGS}

install: $(DEEPSPEECH_BIN)
	install -d ${PREFIX}/lib
	install -m 0644 ${TFDIR}/bazel-bin/native_client/libdeepspeech.so ${PREFIX}/lib/
	install -d ${PREFIX}/include
	install -m 0644 deepspeech.h ${PREFIX}/include
	install -d ${PREFIX}/bin
	install -m 0755 deepspeech ${PREFIX}/bin/

uninstall:
	rm -f ${PREFIX}/bin/deepspeech
	rmdir --ignore-fail-on-non-empty ${PREFIX}/bin
	rm -f ${PREFIX}/lib/libdeepspeech.so
	rmdir --ignore-fail-on-non-empty ${PREFIX}/lib

print-toolchain:
	@echo $(TOOLCHAIN)
