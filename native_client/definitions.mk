NC_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TARGET ?= host
TFDIR  ?= $(abspath $(NC_DIR)/../../tensorflow)
PREFIX ?= /usr/local

ifeq ($(TARGET),host)
TOOLCHAIN       :=
CFLAGS          := `pkg-config --cflags sox`
LDFLAGS         := `pkg-config --libs sox`
PYTHON_PACKAGES := numpy
endif

ifeq ($(TARGET),rpi3)
TOOLCHAIN   ?= ${TFDIR}/bazel-$(shell basename "${TFDIR}")/external/GccArmRpi/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-
RASPBIAN    ?= $(abspath $(NC_DIR)/../multistrap-raspbian-jessie)
CFLAGS      := -isystem $(RASPBIAN)/usr/include -isystem $(RASPBIAN)/usr/include/arm-linux-gnueabihf
LDFLAGS     := -Wl,-rpath-link,$(RASPBIAN)/lib/arm-linux-gnueabihf -Wl,-rpath-link,$(RASPBIAN)/usr/lib/arm-linux-gnueabihf/ $(RASPBIAN)/usr/lib/arm-linux-gnueabihf/libsox.so

PYVER := $(shell python -c "import platform; maj, min, _ = platform.python_version_tuple(); print(maj+'.'+min);")
PYTHON_PACKAGES      :=
NUMPY_INCLUDE        := NUMPY_INCLUDE=$(RASPBIAN)/usr/include/python$(PYVER)/
PYTHON_PLATFORM_NAME := --plat-name linux_armv7l
NODE_PLATFORM_TARGET := --target_arch=arm --target_platform=linux
endif

OS := $(shell uname -s)
CFLAGS  += $(EXTRA_CFLAGS)
LIBS    := -ldeepspeech -ldeepspeech_utils -ltensorflow_cc $(EXTRA_LIBS)
LDFLAGS += -Wl,-rpath,. -L${TFDIR}/bazel-bin/tensorflow -L${TFDIR}/bazel-bin/native_client $(EXTRA_LDFLAGS) $(LIBS)

AS      := $(TOOLCHAIN)as
CC      := $(TOOLCHAIN)gcc
CXX     := $(TOOLCHAIN)c++
LD      := $(TOOLCHAIN)ld

META_LD_LIBRARY_PATH := LD_LIBRARY_PATH
ifeq ($(OS),Darwin)
META_LD_LIBRARY_PATH := DYLD_LIBRARY_PATH
endif
