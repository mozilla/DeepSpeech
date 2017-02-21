# Description: Deepspeech Serving Client.

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

py_binary(
    name = "deepspeech_client",
    srcs = [
        "deepspeech_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
