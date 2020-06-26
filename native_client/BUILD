# Description: Deepspeech native client library.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

load(
    "@org_tensorflow//tensorflow/lite:build_def.bzl",
    "tflite_copts",
    "tflite_linkopts",
)

config_setting(
    name = "tflite",
    define_values = {
        "runtime": "tflite",
    },
)

genrule(
    name = "workspace_status",
    outs = ["workspace_status.cc"],
    cmd = "$(location :gen_workspace_status.sh) >$@",
    local = 1,
    stamp = 1,
    tools = [":gen_workspace_status.sh"],
)


OPENFST_SOURCES_PLATFORM = select({
    "//tensorflow:windows": glob(["ctcdecode/third_party/openfst-1.6.9-win/src/lib/*.cc"]),
    "//conditions:default": glob(["ctcdecode/third_party/openfst-1.6.7/src/lib/*.cc"]),
})

OPENFST_INCLUDES_PLATFORM = select({
    "//tensorflow:windows": ["ctcdecode/third_party/openfst-1.6.9-win/src/include"],
    "//conditions:default": ["ctcdecode/third_party/openfst-1.6.7/src/include"],
})

LINUX_LINKOPTS = [
    "-ldl",
    "-pthread",
    "-Wl,-Bsymbolic",
    "-Wl,-Bsymbolic-functions",
    "-Wl,-export-dynamic",
]

cc_library(
    name = "kenlm",
    srcs = glob([
        "kenlm/lm/*.cc",
        "kenlm/util/*.cc",
        "kenlm/util/double-conversion/*.cc",
        "kenlm/util/double-conversion/*.h",
    ],
    exclude = [
        "kenlm/*/*test.cc",
        "kenlm/*/*main.cc",
    ],),
    hdrs = glob([
        "kenlm/lm/*.hh",
        "kenlm/util/*.hh",
    ]),
    copts = ["-std=c++11"],
    defines = ["KENLM_MAX_ORDER=6"],
    includes = ["kenlm"],
)

cc_library(
    name = "decoder",
    srcs = [
        "ctcdecode/ctc_beam_search_decoder.cpp",
        "ctcdecode/decoder_utils.cpp",
        "ctcdecode/decoder_utils.h",
        "ctcdecode/scorer.cpp",
        "ctcdecode/path_trie.cpp",
        "ctcdecode/path_trie.h",
    ] + OPENFST_SOURCES_PLATFORM,
    hdrs = [
        "ctcdecode/ctc_beam_search_decoder.h",
        "ctcdecode/scorer.h",
    ],
    includes = [
        ".",
        "ctcdecode/third_party/ThreadPool",
    ] + OPENFST_INCLUDES_PLATFORM,
    deps = [":kenlm"]
)

tf_cc_shared_object(
    name = "libdeepspeech.so",
    srcs = [
        "alphabet.h",
        "deepspeech.cc",
        "deepspeech.h",
        "deepspeech_errors.cc",
        "modelstate.cc",
        "modelstate.h",
        "workspace_status.cc",
        "workspace_status.h",
    ] + select({
        "//native_client:tflite": [
            "tflitemodelstate.h",
            "tflitemodelstate.cc",
        ],
        "//conditions:default": [
            "tfmodelstate.h",
            "tfmodelstate.cc",
        ],
    }),
    copts = select({
        # -fvisibility=hidden is not required on Windows, MSCV hides all declarations by default
        "//tensorflow:windows": ["/w"],
        # -Wno-sign-compare to silent a lot of warnings from tensorflow itself,
        # which makes it harder to see our own warnings
        "//conditions:default": [
            "-Wno-sign-compare",
            "-fvisibility=hidden",
        ],
    }) + select({
        "//native_client:tflite": ["-DUSE_TFLITE"],
        "//conditions:default": ["-UUSE_TFLITE"],
    }) + tflite_copts(),
    linkopts = select({
        "//tensorflow:macos": [],
        "//tensorflow:linux_x86_64": LINUX_LINKOPTS,
        "//tensorflow:rpi3": LINUX_LINKOPTS,
        "//tensorflow:rpi3-armv8": LINUX_LINKOPTS,
        "//tensorflow:windows": [],
        "//conditions:default": [],
    }) + tflite_linkopts(),
    deps = select({
        "//native_client:tflite": [
            "//tensorflow/lite/kernels:builtin_ops",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:direct_session",
            "//third_party/eigen3",
            #"//tensorflow/core:all_kernels",
            ### => Trying to be more fine-grained
            ### Use bin/ops_in_graph.py to list all the ops used by a frozen graph.
            ### CPU only build, libdeepspeech.so file size reduced by ~50%
            "//tensorflow/core/kernels:spectrogram_op",  # AudioSpectrogram
            "//tensorflow/core/kernels:bias_op",  # BiasAdd
            "//tensorflow/core/kernels:cast_op",  # Cast
            "//tensorflow/core/kernels:concat_op",  # ConcatV2
            "//tensorflow/core/kernels:constant_op",  # Const, Placeholder
            "//tensorflow/core/kernels:shape_ops",  # ExpandDims, Shape
            "//tensorflow/core/kernels:gather_nd_op",  # GatherNd
            "//tensorflow/core/kernels:identity_op",  # Identity
            "//tensorflow/core/kernels:immutable_constant_op",  # ImmutableConst (used in memmapped models)
            "//tensorflow/core/kernels:deepspeech_cwise_ops",  # Less, Minimum, Mul
            "//tensorflow/core/kernels:matmul_op",  # MatMul
            "//tensorflow/core/kernels:reduction_ops",  # Max
            "//tensorflow/core/kernels:mfcc_op",  # Mfcc
            "//tensorflow/core/kernels:no_op",  # NoOp
            "//tensorflow/core/kernels:pack_op",  # Pack
            "//tensorflow/core/kernels:sequence_ops",  # Range
            "//tensorflow/core/kernels:relu_op",  # Relu
            "//tensorflow/core/kernels:reshape_op",  # Reshape
            "//tensorflow/core/kernels:softmax_op",  # Softmax
            "//tensorflow/core/kernels:tile_ops",  # Tile
            "//tensorflow/core/kernels:transpose_op",  # Transpose
            "//tensorflow/core/kernels:rnn_ops",  # BlockLSTM
            # And we also need the op libs for these ops used in the model:
            "//tensorflow/core:audio_ops_op_lib",  # AudioSpectrogram, Mfcc
            "//tensorflow/core:rnn_ops_op_lib",  # BlockLSTM
            "//tensorflow/core:math_ops_op_lib",  # Cast, Less, Max, MatMul, Minimum, Range
            "//tensorflow/core:array_ops_op_lib",  # ConcatV2, Const, ExpandDims, Fill, GatherNd, Identity, Pack, Placeholder, Reshape, Tile, Transpose
            "//tensorflow/core:no_op_op_lib",  # NoOp
            "//tensorflow/core:nn_ops_op_lib",  # Relu, Softmax, BiasAdd
            # And op libs for these ops brought in by dependencies of dependencies to silence unknown OpKernel warnings:
            "//tensorflow/core:dataset_ops_op_lib",  # UnwrapDatasetVariant, WrapDatasetVariant
            "//tensorflow/core:sendrecv_ops_op_lib",  # _HostRecv, _HostSend, _Recv, _Send
        ],
    }) + if_cuda([
        "//tensorflow/core:core",
    ]) + [":decoder"],
)

genrule(
    name = "libdeepspeech_so_dsym",
    srcs = [":libdeepspeech.so"],
    outs = ["libdeepspeech.so.dSYM"],
    output_to_bindir = True,
    cmd = "dsymutil $(location :libdeepspeech.so) -o $@"
)

cc_binary(
    name = "enumerate_kenlm_vocabulary",
    srcs = [
        "enumerate_kenlm_vocabulary.cpp",
    ],
    deps = [":kenlm"],
    copts = ["-std=c++11"],
)

cc_binary(
    name = "trie_load",
    srcs = [
        "alphabet.h",
        "trie_load.cc",
    ],
    copts = ["-std=c++11"],
    linkopts = [
        "-lm",
        "-ldl",
        "-pthread",
    ],
    deps = [":decoder"],
)
