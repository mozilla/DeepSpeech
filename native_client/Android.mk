LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE    := deepspeech-prebuilt
LOCAL_SRC_FILES := $(TFDIR)/bazel-bin/native_client/libdeepspeech.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_CPP_EXTENSION    := .cc .cxx .cpp
LOCAL_MODULE           := deepspeech
LOCAL_SRC_FILES        := client.cc
LOCAL_SHARED_LIBRARIES := deepspeech-prebuilt
LOCAL_LDFLAGS          := -Wl,--no-as-needed
include $(BUILD_EXECUTABLE)
