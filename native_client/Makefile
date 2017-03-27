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

TARGET ?= host
TFDIR  ?= ../../tensorflow
CXX    ?= c++

ifeq ($(TARGET),host)
TOOLCHAIN :=
CFLAGS    := `pkg-config --cflags sox`
LDFLAGS   := `pkg-config --libs sox`
endif

ifeq ($(TARGET),rpi3)
TOOLCHAIN   ?= ${TFDIR}/bazel-tensorflow/external/GccArmRpi/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
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
	rm -f deepspeech

deepspeech: client.cc
	$(TOOLCHAIN)$(CXX) -o deepspeech $(CFLAGS) client.cc $(LDFLAGS)

run: deepspeech
	${META_LD_LIBRARY_PATH}=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${${META_LD_LIBRARY_PATH}} ./deepspeech ${ARGS}
