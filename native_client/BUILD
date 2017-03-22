# Description: Deepspeech native client library.

cc_library(
    name = "deepspeech",
    srcs = ["deepspeech.cc"],
    hdrs = ["deepspeech.h"],
    deps = ["//tensorflow/core:tensorflow",
            ":c_speech_features"]
)

cc_library(
    name = "kissfft",
    srcs = ["kiss_fft130/kiss_fft.c",
            "kiss_fft130/tools/kiss_fftr.c"],
    hdrs = ["kiss_fft130/kiss_fft.h",
            "kiss_fft130/_kiss_fft_guts.h",
            "kiss_fft130/tools/kiss_fftr.h"],
    includes = ["kiss_fft130"]
)

# Note, we need gnu99 as opposed to c99 as the latter doesn't define M_PI
cc_library(
    name="c_speech_features",
    srcs = ["c_speech_features/c_speech_features.c"],
    hdrs = ["c_speech_features/c_speech_features.h"],
    includes = ["c_speech_features"],
    deps = [":kissfft"],
    copts = ["-std=gnu99"]
)
