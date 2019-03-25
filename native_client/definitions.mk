NC_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TARGET    ?= host
TFDIR     ?= $(abspath $(NC_DIR)/../../tensorflow)
PREFIX    ?= /usr/local
SO_SEARCH ?= $(TFDIR)/bazel-bin/

TOOL_AS   := as
TOOL_CC   := gcc
TOOL_CXX  := c++
TOOL_LD   := ld
TOOL_LDD  := ldd

DEEPSPEECH_BIN       := deepspeech
CFLAGS_DEEPSPEECH    := -std=c++11 -o $(DEEPSPEECH_BIN)
LINK_DEEPSPEECH      := -ldeepspeech
LINK_PATH_DEEPSPEECH := -L${TFDIR}/bazel-bin/native_client

ifeq ($(TARGET),host)
TOOLCHAIN       :=
CFLAGS          :=
CXXFLAGS        :=
LDFLAGS         :=
SOX_CFLAGS      := `pkg-config --cflags sox`
SOX_LDFLAGS     := `pkg-config --libs sox`
PYTHON_PACKAGES := numpy${NUMPY_BUILD_VERSION}
ifeq ($(OS),Linux)
PYTHON_PLATFORM_NAME := --plat-name manylinux1_x86_64
endif
endif

ifeq ($(TARGET),host-win)
DEEPSPEECH_BIN  := deepspeech.exe
TOOLCHAIN := '$(VCINSTALLDIR)\bin\amd64\'
TOOL_CC   := cl.exe
TOOL_CXX  := cl.exe
TOOL_LD   := link.exe
LINK_DEEPSPEECH      := $(TFDIR)\bazel-bin\native_client\libdeepspeech.so.if.lib
LINK_PATH_DEEPSPEECH :=
CFLAGS_DEEPSPEECH    := -nologo -Fe$(DEEPSPEECH_BIN)
SOX_CFLAGS      :=
SOX_LDFLAGS     :=
endif

ifeq ($(TARGET),rpi3)
TOOLCHAIN   ?= ${TFDIR}/bazel-$(shell basename "${TFDIR}")/external/LinaroArmGcc72/bin/arm-linux-gnueabihf-
RASPBIAN    ?= $(abspath $(NC_DIR)/../multistrap-raspbian-stretch)
CFLAGS      := -march=armv7-a -mtune=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard -D_GLIBCXX_USE_CXX11_ABI=0 --sysroot $(RASPBIAN)
CXXFLAGS    := $(CXXFLAGS)
LDFLAGS     := -Wl,-rpath-link,$(RASPBIAN)/lib/arm-linux-gnueabihf/ -Wl,-rpath-link,$(RASPBIAN)/usr/lib/arm-linux-gnueabihf/

SOX_CFLAGS  := -I$(RASPBIAN)/usr/include
SOX_LDFLAGS := $(RASPBIAN)/usr/lib/arm-linux-gnueabihf/libsox.so

PYVER := $(shell python -c "import platform; maj, min, _ = platform.python_version_tuple(); print(maj+'.'+min);")
PYTHON_PACKAGES      :=
PYTHON_PATH          := PYTHONPATH=$(RASPBIAN)/usr/lib/python$(PYVER)/:$(RASPBIAN)/usr/lib/python$(PYVER)/plat-arm-linux-gnueabihf/:$(RASPBIAN)/usr/lib/python3/dist-packages/
NUMPY_INCLUDE        := NUMPY_INCLUDE=$(RASPBIAN)/usr/include/python3.5/
PYTHON_PLATFORM_NAME := --plat-name linux_armv7l
NODE_PLATFORM_TARGET := --target_arch=arm --target_platform=linux
TOOLCHAIN_LDD_OPTS   := --root $(RASPBIAN)/
endif # ($(TARGET),rpi3)

ifeq ($(TARGET),rpi3-armv8)
TOOLCHAIN   ?= ${TFDIR}/bazel-$(shell basename "${TFDIR}")/external/LinaroAarch64Gcc72/bin/aarch64-linux-gnu-
RASPBIAN    ?= $(abspath $(NC_DIR)/../multistrap-raspbian64-stretch)
CFLAGS      := -march=armv8-a -mtune=cortex-a53 -D_GLIBCXX_USE_CXX11_ABI=0 --sysroot $(RASPBIAN)
CXXFLAGS    := $(CFLAGS)
LDFLAGS     := -Wl,-rpath-link,$(RASPBIAN)/lib/aarch64-linux-gnu/ -Wl,-rpath-link,$(RASPBIAN)/usr/lib/aarch64-linux-gnu/

SOX_CFLAGS  := -I$(RASPBIAN)/usr/include
SOX_LDFLAGS := $(RASPBIAN)/usr/lib/aarch64-linux-gnu/libsox.so

PYVER := $(shell python -c "import platform; maj, min, _ = platform.python_version_tuple(); print(maj+'.'+min);")
PYTHON_PACKAGES      :=
PYTHON_PATH          := PYTHONPATH=$(RASPBIAN)/usr/lib/python$(PYVER)/:$(RASPBIAN)/usr/lib/python$(PYVER)/plat-aarch64-linux-gnu/:$(RASPBIAN)/usr/lib/python3/dist-packages/
NUMPY_INCLUDE        := NUMPY_INCLUDE=$(RASPBIAN)/usr/include/python3.5/
PYTHON_PLATFORM_NAME := --plat-name linux_aarch64
NODE_PLATFORM_TARGET := --target_arch=arm64 --target_platform=linux
TOOLCHAIN_LDD_OPTS   := --root $(RASPBIAN)/
endif # ($(TARGET),rpi3-armv8)

OS      := $(shell uname -s)

# -Wl,--no-as-needed is required to force linker not to evict libs it thinks we
# dont need ; will fail the build on OSX because that option does not exists
ifeq ($(OS),Linux)
LDFLAGS_NEEDED := -Wl,--no-as-needed
LDFLAGS_RPATH  := -Wl,-rpath,\$$ORIGIN
endif
ifeq ($(OS),Darwin)
CXXFLAGS       += -stdlib=libc++ -mmacosx-version-min=10.10
LDFLAGS_NEEDED := -stdlib=libc++ -mmacosx-version-min=10.10
LDFLAGS_RPATH  := -Wl,-rpath,@executable_path
endif

CFLAGS   += $(EXTRA_CFLAGS)
CXXFLAGS += $(EXTRA_CXXFLAGS)
LIBS     := $(LINK_DEEPSPEECH) $(EXTRA_LIBS)
LDFLAGS_DIRS := $(LINK_PATH_DEEPSPEECH) $(EXTRA_LDFLAGS)
LDFLAGS  += $(LDFLAGS_NEEDED) $(LDFLAGS_RPATH) $(LDFLAGS_DIRS) $(LIBS)

AS      := $(TOOLCHAIN)$(TOOL_AS)
CC      := $(TOOLCHAIN)$(TOOL_CC)
CXX     := $(TOOLCHAIN)$(TOOL_CXX)
LD      := $(TOOLCHAIN)$(TOOL_LD)
LDD     := $(TOOLCHAIN)$(TOOL_LDD) $(TOOLCHAIN_LDD_OPTS)

RPATH_PYTHON         := '-Wl,-rpath,\$$ORIGIN/lib/' $(LDFLAGS_RPATH)
RPATH_NODEJS         := '-Wl,-rpath,$$\$$ORIGIN/../'
META_LD_LIBRARY_PATH := LD_LIBRARY_PATH
ifeq ($(OS),Darwin)
META_LD_LIBRARY_PATH := DYLD_LIBRARY_PATH
RPATH_PYTHON         := '-Wl,-rpath,@loader_path/lib/' $(LDFLAGS_RPATH)
RPATH_NODEJS         := '-Wl,-rpath,@loader_path/../'
endif

# Takes care of looking into bindings built (SRC_FILE, can contain a wildcard)
# for missing dependencies and copying those dependencies into the
# TARGET_LIB_DIR. If supplied, MANIFEST_IN will be echo'ed to a list of
# 'include x.so'.
#
# On OSX systems, this will also take care of calling install_name_tool to set
# proper path for those dependencies, using @rpath/lib.
define copy_missing_libs
    SRC_FILE=$(1); \
    TARGET_LIB_DIR=$(2); \
    MANIFEST_IN=$(3); \
    echo "Analyzing $$SRC_FILE copying missing libs to $$SRC_FILE"; \
    echo "Maybe outputting to $$MANIFEST_IN"; \
    \
    (mkdir $$TARGET_LIB_DIR || true); \
    missing_libs=""; \
    for lib in $$SRC_FILE; do \
        if [ "$(OS)" = "Darwin" ]; then \
            new_missing="$$( (for f in $$(otool -L $$lib 2>/dev/null | tail -n +2 | awk '{ print $$1 }' | grep -v '$$lib'); do ls -hal $$f; done;) 2>&1 | grep 'No such' | cut -d':' -f2 | xargs basename -a)"; \
            missing_libs="$$missing_libs $$new_missing"; \
	elif [ "$(OS)" = "${TC_MSYS_VERSION}" ]; then \
            missing_libs="libdeepspeech.so"; \
        else \
            missing_libs="$$missing_libs $$($(LDD) $$lib | grep 'not found' | awk '{ print $$1 }')"; \
        fi; \
    done; \
    \
    for missing in $$missing_libs; do \
        find $(SO_SEARCH) -type f -name "$$missing" -exec cp {} $$TARGET_LIB_DIR \; ; \
        if [ ! -z "$$MANIFEST_IN" ]; then \
            echo "include $$TARGET_LIB_DIR/$$missing" >> $$MANIFEST_IN; \
        fi; \
    done; \
    \
    if [ "$(OS)" = "Darwin" ]; then \
        for lib in $$SRC_FILE; do \
            for dep in $$( (for f in $$(otool -L $$lib 2>/dev/null | tail -n +2 | awk '{ print $$1 }' | grep -v '$$lib'); do ls -hal $$f; done;) 2>&1 | grep 'No such' | cut -d':' -f2 ); do \
                dep_basename=$$(basename "$$dep"); \
                install_name_tool -change "$$dep" "@rpath/$$dep_basename" "$$lib"; \
            done; \
        done; \
    fi;
endef
