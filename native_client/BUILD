# Description: Deepspeech native client library.

load("//tensorflow:tensorflow.bzl",
     "if_linux_x86_64")

cc_library(
    name = "deepspeech",
    srcs = ["deepspeech.cc", "alphabet.h"],
    hdrs = ["deepspeech.h"],
    deps = ["//tensorflow/core:core",
            ":deepspeech_utils"],
)

# We have a single rule including c_speech_features and kissfft here as Bazel
# doesn't support static linking in library targets.

cc_library(
    name = "deepspeech_utils",
    srcs = ["deepspeech_utils.cc",
            "c_speech_features/c_speech_features.c",
            "kiss_fft130/kiss_fft.c",
            "kiss_fft130/tools/kiss_fftr.c"],
    hdrs = ["deepspeech_utils.h",
            "c_speech_features/c_speech_features.h",
            "c_speech_features/c_speech_features_config.h",
            "kiss_fft130/kiss_fft.h",
            "kiss_fft130/_kiss_fft_guts.h",
            "kiss_fft130/tools/kiss_fftr.h"],
    includes = ["c_speech_features",
                "kiss_fft130"],

    # fma/avx/avx2 enabled in gcc cause significant performance decreases in
    # c_speech_features and so are force-disabled.
    copts = [] + if_linux_x86_64(["-mno-fma", "-mno-avx", "-mno-avx2"]),
    nocopts = "(-fstack-protector|-fno-omit-frame-pointer)",
)


cc_library(
    name = "ctc_decoder_with_kenlm",
    srcs = ["beam_search.cc",
            "alphabet.h",
            "trie_node.h"] +
           glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]),
    includes = ["kenlm"],
    defines = ["KENLM_MAX_ORDER=6"],
    deps = ["//tensorflow/core:core",
            "//tensorflow/core/util/ctc",
            "//third_party/eigen3",
    ],
)

cc_binary(
    name = "generate_trie",
    srcs = [
        "generate_trie.cpp",
        "trie_node.h",
        "alphabet.h",
    ] + glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]),
    includes = ["kenlm"],
    copts = ['-std=c++11'],
    linkopts = ['-lm'],
    defines = ["KENLM_MAX_ORDER=6"],
)
