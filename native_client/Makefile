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

TARGET ?= host
TFDIR  ?= ../../tensorflow
CXX    ?= c++
PREFIX ?= /usr/local

ifeq ($(TARGET),host)
TOOLCHAIN :=
CFLAGS    := `pkg-config --cflags sox`
LDFLAGS   := `pkg-config --libs sox`
endif

ifeq ($(TARGET),rpi3)
TOOLCHAIN   ?= ${TFDIR}/bazel-$(shell basename "${TFDIR}")/external/GccArmRpi/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
RASPBIAN    ?= ../multistrap-raspbian-jessie
CFLAGS      := -isystem $(RASPBIAN)/usr/include -L$(RASPBIAN)/lib/arm-linux-gnueabihf/ -L$(RASPBIAN)/usr/lib/arm-linux-gnueabihf/
LDFLAGS     := -Wl,-rpath-link,$(RASPBIAN)/lib/arm-linux-gnueabihf -Wl,-rpath-link,$(RASPBIAN)/usr/lib/arm-linux-gnueabihf/ -lsox
endif

OS := $(shell uname -s)
CFLAGS  += -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client $(EXTRA_CFLAGS)
LDFLAGS += -ltensorflow -ldeepspeech -lc_speech_features -lkissfft $(EXTRA_LDFLAGS)

META_LD_LIBRARY_PATH := LD_LIBRARY_PATH
ifeq ($(OS),Darwin)
META_LD_LIBRARY_PATH := DYLD_LIBRARY_PATH
endif

default: deepspeech

clean:
	rm -rf build dist deepspeech.egg-info
	rm -f deepspeech setup.cfg python/deepspeech_wrap.cpp python/deepspeech.py

deepspeech: client.cc
	$(TOOLCHAIN)$(CXX) -o deepspeech $(CFLAGS) client.cc $(LDFLAGS)

setup.cfg: setup.cfg.in
	sed -e 's:@LIBDIRS@:${TFDIR}/bazel-bin/tensorflow\:${TFDIR}/bazel-bin/native_client:g' setup.cfg.in > setup.cfg

bindings: setup.cfg
	python ./setup.py bdist_wheel

run: deepspeech
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} ./deepspeech ${ARGS}

install:
	install -d ${PREFIX}/lib
	install -m 0644 ${TFDIR}/bazel-bin/tensorflow/libtensorflow.so ${PREFIX}/lib/
	install -m 0644 ${TFDIR}/bazel-bin/native_client/libkissfft.so ${PREFIX}/lib/
	install -m 0644 ${TFDIR}/bazel-bin/native_client/libc_speech_features.so ${PREFIX}/lib/
	install -m 0644 ${TFDIR}/bazel-bin/native_client/libdeepspeech.so ${PREFIX}/lib/

uninstall:
	rm -f ${PREFIX}/lib/libtensorflow.so
	rm -f ${PREFIX}/lib/libkissfft.so
	rm -f ${PREFIX}/lib/libc_speech_features.so
	rm -f ${PREFIX}/lib/libdeepspeech.so
