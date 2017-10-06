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
CFLAGS      := -isystem $(RASPBIAN)/usr/include -isystem $(RASPBIAN)/usr/include/arm-linux-gnueabihf
LDFLAGS     := -Wl,-rpath-link,$(RASPBIAN)/lib/arm-linux-gnueabihf -Wl,-rpath-link,$(RASPBIAN)/usr/lib/arm-linux-gnueabihf/ $(RASPBIAN)/usr/lib/arm-linux-gnueabihf/libsox.so
endif

OS := $(shell uname -s)
CFLAGS  += $(EXTRA_CFLAGS)
LIBS    := -ldeepspeech -ldeepspeech_utils -ltensorflow_cc $(EXTRA_LIBS)
LDFLAGS += -Wl,-rpath,. -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client $(EXTRA_LDFLAGS) $(LIBS)

META_LD_LIBRARY_PATH := LD_LIBRARY_PATH
ifeq ($(OS),Darwin)
META_LD_LIBRARY_PATH := DYLD_LIBRARY_PATH
endif
