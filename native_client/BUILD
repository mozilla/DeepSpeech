# Description: Deepspeech native client library.

load("//tensorflow:tensorflow.bzl",
     "if_linux_x86_64", "tf_cc_shared_object")

load("//tensorflow/compiler/aot:tfcompile.bzl",
     "tf_library")

load(":deepspeech.bzl", "if_native_model")

config_setting(
    name = "ds_native_model",
    values = {
        "define": "DS_NATIVE_MODEL=1"
    }
)

tf_library(
    name = "deepspeech_model",
    cpp_class = "DeepSpeech::nativeModel",
    # We don't need tests or benchmark binaries
    gen_test=False, gen_benchmark=False,
    # graph and config will be generated at build time thanks to the matching
    # genrule.
    graph = "tfcompile.model.pb",
    config = "tfcompile.config.pbtxt",
    # This depends on //tensorflow:rpi3 condition defined in mozilla/tensorflow
    tfcompile_flags = select({
        "//tensorflow:rpi3": str('--target_triple="armv6-linux-gnueabihf" --target_cpu="cortex-a53" --target_features="+neon-fp-armv8"'),
        "//conditions:default": str('')
    }),
)

genrule(
    name = "tfcompile.config",
    srcs = ["tfcompile.config.pbtxt.src"],
    outs = ["tfcompile.config.pbtxt"],
    cmd = "$(location :model_size.sh) $(SRCS) $(DS_MODEL_TIMESTEPS) $(DS_MODEL_FRAMESIZE) >$@",
    tools = [":model_size.sh"]
)

genrule(
    name = "tfcompile.model",
    outs = ["tfcompile.model.pb"],
    cmd = "cp $(DS_MODEL_FILE) $@"
)

cc_library(
    name = "deepspeech",
    srcs = ["deepspeech.cc", "alphabet.h", "beam_search.h", "trie_node.h"] +
           glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]) +
           glob(["boost_locale/**/*.hpp"]),
    hdrs = ["deepspeech.h"],
    # -Wno-sign-compare to silent a lot of warnings from tensorflow itself,
    # which makes it harder to see our own warnings
    copts = ["-std=c++11", "-Wno-sign-compare"] + if_native_model([
        "-DDS_MODEL_TIMESTEPS=$(DS_MODEL_TIMESTEPS)",
        "-DDS_NATIVE_MODEL=1",
    ]),
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-install_name,@rpath/libdeepspeech.so"
	],
	"//conditions:default": []
    }),
    deps = [
        "//tensorflow/core:core",
        "//tensorflow/core/util/ctc",
        "//third_party/eigen3",
        ":deepspeech_utils"
    ] + if_native_model([":deepspeech_model", "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function"]),
    includes = ["kenlm", "boost_locale"],
    defines = ["KENLM_MAX_ORDER=6"],
)

# We have a single rule including c_speech_features and kissfft here as Bazel
# doesn't support static linking in library targets.

cc_library(
    name = "deepspeech_utils",
    srcs = ["deepspeech_utils.cc",
            "c_speech_features/c_speech_features.c",
            "kiss_fft130/kiss_fft.c",
            "kiss_fft130/tools/kiss_fftr.c",
            "c_speech_features/c_speech_features.h",
            "c_speech_features/c_speech_features_config.h",
            "kiss_fft130/kiss_fft.h",
            "kiss_fft130/_kiss_fft_guts.h",
            "kiss_fft130/tools/kiss_fftr.h"],
    hdrs = ["deepspeech_utils.h"],
    includes = ["c_speech_features",
                "kiss_fft130"],

    # fma/avx/avx2 enabled in gcc cause significant performance decreases in
    # c_speech_features and so are force-disabled.
    copts = [] + if_linux_x86_64(["-mno-fma", "-mno-avx", "-mno-avx2"]),
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-install_name,@rpath/libdeepspeech_utils.so"
	],
	"//conditions:default": []
    }),
    nocopts = "(-fstack-protector|-fno-omit-frame-pointer)",
)


tf_cc_shared_object(
    name = "libctc_decoder_with_kenlm.so",
    srcs = [
            "beam_search.cc",
            "beam_search.h",
            "alphabet.h",
            "trie_node.h"
           ] +
           glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]) +
           glob(["boost_locale/**/*.hpp"]),
    includes = ["kenlm", "boost_locale"],
    copts = ["-std=c++11"],
    defines = ["KENLM_MAX_ORDER=6"],
    deps = ["//tensorflow/core:framework_headers_lib",
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
           ] +
           glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]) +
           glob(["boost_locale/**/*.hpp"]),
    includes = ["kenlm", "boost_locale"],
    copts = ["-std=c++11"],
    linkopts = ["-lm"],
    defines = ["KENLM_MAX_ORDER=6"],
)
