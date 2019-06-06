# Description: Deepspeech native client library.

load("@org_tensorflow//tensorflow:tensorflow.bzl",
     "tf_cc_shared_object", "if_cuda")

load("@org_tensorflow//tensorflow/lite:build_def.bzl",
     "tflite_copts", "tflite_linkopts")

config_setting(
    name = "tflite",
    define_values = {
        "runtime": "tflite"
    },
)

genrule(
    name = "ds_git_version",
    outs = ["ds_version.h"],
    cmd = "$(location :ds_git_version.sh) >$@",
    tools = [":ds_git_version.sh"],
    local = 1,
)

genrule(
    name = "ds_graph_version",
    outs = ["ds_graph_version.h"],
    cmd = "$(location :ds_graph_version.sh) >$@",
    tools = [":ds_graph_version.sh"],
    local = 1,
)

KENLM_SOURCES = glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
                      "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"],
                     exclude = ["kenlm/*/*test.cc", "kenlm/*/*main.cc"])

KENLM_INCLUDES = [
    "kenlm",
]

OPENFST_SOURCES_PLATFORM = select({ 
        "//tensorflow:windows": glob(["ctcdecode/third_party/openfst-1.6.9-win/src/lib/*.cc"]),
        "//conditions:default": glob(["ctcdecode/third_party/openfst-1.6.7/src/lib/*.cc"]),
})

DECODER_SOURCES = glob([
    "ctcdecode/*.h",
    "ctcdecode/*.cpp",  
]) + OPENFST_SOURCES_PLATFORM + KENLM_SOURCES

OPENFST_INCLUDES_PLATFORM = select({ 
        "//tensorflow:windows": ["ctcdecode/third_party/openfst-1.6.9-win/src/include"],
        "//conditions:default": ["ctcdecode/third_party/openfst-1.6.7/src/include"],
})

DECODER_INCLUDES = [
    ".",
    "ctcdecode/third_party/ThreadPool",   
] + OPENFST_INCLUDES_PLATFORM + KENLM_INCLUDES

LINUX_LINKOPTS = [
    "-ldl",
    "-pthread",
    "-Wl,-Bsymbolic",
    "-Wl,-Bsymbolic-functions",
    "-Wl,-export-dynamic"
]

tf_cc_shared_object(
    name = "libdeepspeech.so",
    srcs = ["deepspeech.cc",
            "deepspeech.h",
            "alphabet.h",
            "modelstate.h",
            "modelstate.cc",
            "ds_version.h",
            "ds_graph_version.h"] +
            DECODER_SOURCES +
            select({
                "//native_client:tflite": [
                    "tflitemodelstate.h",
                    "tflitemodelstate.cc"
                ],
                "//conditions:default": [
                    "tfmodelstate.h",
                    "tfmodelstate.cc"
            ]}),
    copts = select({ 
        # -fvisibility=hidden is not required on Windows, MSCV hides all declarations by default
        "//tensorflow:windows": ["/w"],
        # -Wno-sign-compare to silent a lot of warnings from tensorflow itself,
        # which makes it harder to see our own warnings
        "//conditions:default": ["-Wno-sign-compare", "-fvisibility=hidden"],
    }) + select({
        "//native_client:tflite": [ "-DUSE_TFLITE" ],
        "//conditions:default": [ "-UUSE_TFLITE" ]
    }) + tflite_copts(),
    linkopts = select({
        "//tensorflow:darwin": [],
        "//tensorflow:linux_x86_64": LINUX_LINKOPTS,
        "//tensorflow:rpi3": LINUX_LINKOPTS + ["-l:libstdc++.a"],
        "//tensorflow:rpi3-armv8": LINUX_LINKOPTS + ["-l:libstdc++.a"],
        "//tensorflow:windows": [],
        "//conditions:default": []
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
            "//tensorflow/core/kernels:dense_update_ops",   # Assign (remove once prod model no longer depends on it)
            "//tensorflow/core/kernels:constant_op",        # Placeholder
            "//tensorflow/core/kernels:immutable_constant_op", # ImmutableConst (used in memmapped models)
            "//tensorflow/core/kernels:identity_op",        # Identity
            "//tensorflow/core/kernels:softmax_op",         # Softmax
            "//tensorflow/core/kernels:transpose_op",       # Transpose
            "//tensorflow/core/kernels:reshape_op",         # Reshape
            "//tensorflow/core/kernels:shape_ops",          # Shape
            "//tensorflow/core/kernels:concat_op",          # ConcatV2
            "//tensorflow/core/kernels:relu_op",            # Relu
            "//tensorflow/core/kernels:bias_op",            # BiasAdd
            "//tensorflow/core/kernels:math",               # Range, MatMul
            "//tensorflow/core/kernels:tile_ops",           # Tile
            "//tensorflow/core/kernels:mfcc_op",            # Mfcc
            "//tensorflow/core/kernels:spectrogram_op",     # AudioSpectrogram
            "//tensorflow/core/kernels:strided_slice_op",   # StridedSlice
            "//tensorflow/core/kernels:slice_op",           # Slice, needed by StridedSlice
            "//tensorflow/contrib/rnn:lstm_ops_kernels",    # BlockLSTM
            "//tensorflow/core/kernels:pack_op",            # Pack
            "//tensorflow/core/kernels:gather_nd_op",       # GatherNd
        ],
    }) + if_cuda([
            "//tensorflow/core:core",
    ]),
    includes = DECODER_INCLUDES,
    defines = ["KENLM_MAX_ORDER=6"],
)

cc_binary(
    name = "generate_trie",
    srcs = [
            "generate_trie.cpp",
            "alphabet.h",
           ] + DECODER_SOURCES,
    includes = DECODER_INCLUDES,
    copts = ["-std=c++11"],
    linkopts = ["-lm", "-ldl", "-pthread"],
    defines = ["KENLM_MAX_ORDER=6"],
)

cc_binary(
    name = "trie_load",
    srcs = [
            "trie_load.cc",
            "alphabet.h",
           ] + DECODER_SOURCES,
    includes = DECODER_INCLUDES,
    copts = ["-std=c++11"],
    linkopts = ["-lm", "-ldl", "-pthread"],
    defines = ["KENLM_MAX_ORDER=6"],
)
