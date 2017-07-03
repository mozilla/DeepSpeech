# Description: Deepspeech native client library.

load("//tensorflow:tensorflow.bzl",
     "if_linux_x86_64", "if_cuda")

load("//tensorflow/compiler/aot:tfcompile.bzl",
     "tf_library")

# The same tf_library call from step 2 above.
tf_library(
    # name is used to generate the following underlying build rules:
    # <name>           : cc_library packaging the generated header and object files
    # <name>_test      : cc_test containing a simple test and benchmark
    # <name>_benchmark : cc_binary containing a stand-alone benchmark with minimal deps;
    #                    can be run on a mobile device
    name = "deepspeech_model",
    # cpp_class specifies the name of the generated C++ class, with namespaces allowed.
    # The class will be generated in the given namespace(s), or if no namespaces are
    # given, within the global namespace.
    cpp_class = "DeepSpeech::nativeModel",
    # We don't need tests or benchmark binaries
    gen_test=False, gen_benchmark=False,
    # graph is the input GraphDef proto, by default expected in binary format.  To
    # use the text format instead, just use the ‘.pbtxt’ suffix.  A subgraph will be
    # created from this input graph, with feeds as inputs and fetches as outputs.
    # No Placeholder or Variable ops may exist in this subgraph.
    graph = "tfcompile.model.pb",
    # config is the input Config proto, by default expected in binary format.  To
    # use the text format instead, use the ‘.pbtxt’ suffix.  This is where the
    # feeds and fetches were specified above, in the previous step.
    config = "tfcompile.config.pbtxt",
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
    srcs = ["deepspeech.cc", "alphabet.h"],
    hdrs = ["deepspeech.h"],
    copts = [
        "-DDS_MODEL_TIMESTEPS=$(DS_MODEL_TIMESTEPS)",
    ] + if_cuda([], ["-DTF_HAS_NATIVE_MODEL=1"]),
    deps = [
        "//tensorflow/core:core",
	"//tensorflow/core/util/ctc:ctc_beam_search_lib",
	"//third_party/eigen3",
        ":deepspeech_utils",
    ] + if_cuda([], [":deepspeech_model"]),
    linkopts = [
	"-lpthread",
    ]
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
    nocopts = "(-fstack-protector|-fno-omit-frame-pointer)",
)


cc_library(
    name = "ctc_decoder_with_kenlm",
    srcs = [
            "beam_search.cc",
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
