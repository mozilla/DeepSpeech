# Description: Deepspeech native client library.

load("//tensorflow:tensorflow.bzl",
     "if_x86")

# We have a single rule here as Bazel doesn't support static linking in
# library targets.

cc_library(
    name = "deepspeech",
    srcs = ["deepspeech.cc",
            "c_speech_features/c_speech_features.c",
            "kiss_fft130/kiss_fft.c",
            "kiss_fft130/tools/kiss_fftr.c"],
    hdrs = ["deepspeech.h",
            "c_speech_features/c_speech_features.h",
            "c_speech_features/c_speech_features_config.h",
            "kiss_fft130/kiss_fft.h",
            "kiss_fft130/_kiss_fft_guts.h",
            "kiss_fft130/tools/kiss_fftr.h"],
    includes = ["c_speech_features",
                "kiss_fft130"],
    deps = ["//tensorflow/core:core"],

    # fma/avx/avx2 enabled in gcc cause significant performance decreases in
    # c_speech_features and so are force-disabled.
    copts = [] + if_x86(["-mno-fma", "-mno-avx", "-mno-avx2"]),
    nocopts = "(-fstack-protector|-fno-omit-frame-pointer)",
)
