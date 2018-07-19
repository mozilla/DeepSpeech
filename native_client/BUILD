# Description: Deepspeech native client library.

load("@org_tensorflow//tensorflow:tensorflow.bzl",
     "if_linux_x86_64", "tf_cc_shared_object", "if_cuda")

load("@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl",
     "tf_library")

load(":deepspeech.bzl", "if_native_model")

config_setting(
    name = "ds_native_model",
    values = {
        "define": "DS_NATIVE_MODEL=1"
    }
)

tf_library(
    name = "deepspeech_model_core",
    cpp_class = "DeepSpeech::nativeModel",
    # We don't need tests or benchmark binaries
    gen_test=False, gen_benchmark=False,
    # graph and config will be generated at build time thanks to the matching
    # genrule.
    graph = "tfcompile.model.pb",
    config = "tfcompile.config.pbtxt",
    # This depends on //tensorflow:rpi3 condition defined in mozilla/tensorflow
    tfcompile_flags = select({
        "//tensorflow:rpi3": str('--target_cpu="cortex-a53"'),
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

genrule(
    name = "ds_git_version",
    outs = ["ds_version.h"],
    cmd = "$(location :ds_git_version.sh) >$@",
    tools = [":ds_git_version.sh"]
)

tf_cc_shared_object(
    name = "libdeepspeech.so",
    srcs = ["deepspeech.cc",
            "deepspeech.h",
            "alphabet.h",
            "beam_search.h",
            "trie_node.h",
            "c_speech_features/c_speech_features.cpp",
            "kiss_fft130/kiss_fft.c",
            "kiss_fft130/tools/kiss_fftr.c",
            "c_speech_features/c_speech_features.h",
            "c_speech_features/c_speech_features_config.h",
            "kiss_fft130/kiss_fft.h",
            "kiss_fft130/_kiss_fft_guts.h",
            "kiss_fft130/tools/kiss_fftr.h",
            "ds_version.h"] +
           if_native_model(["deepspeech_model_core.h"]) +
           glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"]) +
           glob(["boost_locale/**/*.hpp"]),
    # -Wno-sign-compare to silent a lot of warnings from tensorflow itself,
    # which makes it harder to see our own warnings
    copts = ["-Wno-sign-compare", "-fvisibility=hidden"] + if_native_model([
        "-DDS_MODEL_TIMESTEPS=$(DS_MODEL_TIMESTEPS)",
        "-DDS_NATIVE_MODEL=1",
    ]),
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core/util/ctc",
        "//third_party/eigen3",
        #"//tensorflow/core:all_kernels",
        ### => Trying to be more fine-grained
        ### Use bin/ops_in_graph.py to list all the ops used by a frozen graph.
        ### CPU only build, libdeepspeech.so file size reduced by ~50%
        "//tensorflow/core/kernels:dense_update_ops",   # Assign
        "//tensorflow/core/kernels:constant_op",        # Const
        "//tensorflow/core/kernels:immutable_constant_op", # ImmutableConst
        "//tensorflow/core/kernels:identity_op",        # Identity
        "//tensorflow/core/kernels:transpose_op",       # Transpose
        "//tensorflow/core/kernels:reshape_op",         # Reshape
        "//tensorflow/core/kernels:shape_ops",          # Shape
        "//tensorflow/core/kernels:concat_op",          # ConcatV2
        "//tensorflow/core/kernels:split_op",           # Split
        "//tensorflow/core/kernels:variable_ops",       # VariableV2
        "//tensorflow/core/kernels:relu_op",            # Relu
        "//tensorflow/core/kernels:bias_op",            # BiasAdd
        "//tensorflow/core/kernels:math",               # Range, MatMul
        "//tensorflow/core/kernels:control_flow_ops",   # Enter
        "//tensorflow/core/kernels:tile_ops",           # Tile
        "//tensorflow/core/kernels:gather_op",          # Gather
        "//tensorflow/contrib/rnn:lstm_ops_kernels"     # BlockLSTM
        #### Needed by production model produced without "--use_seq_length False"
        #"//tensorflow/core/kernels:logging_ops",         # Assert
        #"//tensorflow/core/kernels:reverse_sequence_op", # ReverseSequence
    ] + if_native_model([
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
    ])+ if_cuda([
        "//tensorflow/core:core",
    ]),
    includes = ["kenlm", "boost_locale", "c_speech_features", "kiss_fft130"],
    defines = ["KENLM_MAX_ORDER=6"],
)

tf_cc_shared_object(
    name = "libdeepspeech_model.so",
    deps = [":deepspeech_model_core"]
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
