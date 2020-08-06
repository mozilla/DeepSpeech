LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE    := mozilla_voice_stt-prebuilt
LOCAL_SRC_FILES := $(TFDIR)/bazel-bin/native_client/libmozilla_voice_stt.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_CPP_EXTENSION    := .cc .cxx .cpp
LOCAL_MODULE           := mozilla_voice_stt
LOCAL_SRC_FILES        := client.cc
LOCAL_SHARED_LIBRARIES := mozilla_voice_stt-prebuilt
LOCAL_LDFLAGS          := -Wl,--no-as-needed
include $(BUILD_EXECUTABLE)
