LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE    := aeiou-prebuilt
LOCAL_SRC_FILES := $(TFDIR)/bazel-bin/native_client/libaeiou.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_CPP_EXTENSION    := .cc .cxx .cpp
LOCAL_MODULE           := aeiou
LOCAL_SRC_FILES        := client.cc
LOCAL_SHARED_LIBRARIES := aeiou-prebuilt
LOCAL_LDFLAGS          := -Wl,--no-as-needed
include $(BUILD_EXECUTABLE)
