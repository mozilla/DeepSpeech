# Description: Deepspeech native client library.

cc_library(
    name = "deepspeech",
    srcs = ["deepspeech.cc",
            "c_speech_features/c_speech_features.c",
            "kiss_fft130/kiss_fft.c",
            "kiss_fft130/tools/kiss_fftr.c"],
    hdrs = ["deepspeech.h",
            "c_speech_features/c_speech_features.h",
            "kiss_fft130/kiss_fft.h",
            "kiss_fft130/_kiss_fft_guts.h",
            "kiss_fft130/tools/kiss_fftr.h"],
    includes = ["c_speech_features",
                "kiss_fft130"],
    deps = [
        "//tensorflow/core:tensorflow"
    ]
)
