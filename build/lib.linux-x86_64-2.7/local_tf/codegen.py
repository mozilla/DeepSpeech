# depends on grpcio-tools
#
# This is intended to update the Tensorflow Serving APIs that we extract out of
# tensorflow serving repo for easier reuse. It should not be needed to run that
# except in case of changes to the .proto files with regards to the tensorflow
# serving version you are using.
#
# Running this will regenerate tensorflow/ and tensorflow_serving/. After doing
# so, please commit the changes.

from grpc.tools import protoc

import sys
import os
import shutil
import tempfile

tensorflow_root         = sys.argv[1]
tensorflow_serving_root = sys.argv[2]

if not os.path.isdir(tensorflow_root) or not os.path.isdir(tensorflow_serving_root):
    print "Execution requires git clones of tensorflow and tensorflow serving."
    print "No tensorflow_root (%s) or no tensorflow_serving_root (%s)" % (tensorflow_root, tensorflow_serving_root)
    sys.exit(1)

tmpdir    = tempfile.mkdtemp(suffix='tf_proto', prefix='tmp')
thisdir   = os.path.abspath('.')
distproto = os.path.join(os.path.dirname(protoc.__file__), '_proto') 

tensorflow_serving_apis = os.path.join('tensorflow_serving', 'apis')
model_proto        = os.path.join(tensorflow_serving_root, tensorflow_serving_apis, 'model.proto')
predict_proto      = os.path.join(tensorflow_serving_root, tensorflow_serving_apis, 'predict.proto')
prediction_service = os.path.join(tensorflow_serving_root, tensorflow_serving_apis, 'prediction_service.proto')

tensorflow_core_framework = os.path.join('tensorflow', 'core', 'framework')
tensor_proto          = os.path.join(tensorflow_root, tensorflow_core_framework, 'tensor.proto')
resource_handle_proto = os.path.join(tensorflow_root, tensorflow_core_framework, 'resource_handle.proto')
tensor_shape_proto    = os.path.join(tensorflow_root, tensorflow_core_framework, 'tensor_shape.proto')
types_proto           = os.path.join(tensorflow_root, tensorflow_core_framework, 'types.proto')

protoc.main(
    (
        '',
        '-I%s' % (tensorflow_serving_root), '-I%s' % (os.path.join(tensorflow_serving_root, 'tensorflow')), '-I%s' % (distproto),
        '--python_out=%s' % (tmpdir), '--grpc_python_out=%s' % (tmpdir),
        model_proto, predict_proto, prediction_service,
    )
)

protoc.main(
    (
        '',
        '-I%s' % (tensorflow_root), '-I%s' % (distproto),
        '--python_out=%s' % (tmpdir), '--grpc_python_out=%s' % (tmpdir),
        tensor_proto, resource_handle_proto, tensor_shape_proto, types_proto,
    )
)

def myvisit(a, dir, files):
    if os.path.abspath(dir) == os.path.abspath(tmpdir):
        return

    initpy = os.path.join(dir, '__init__.py')
    try:
        os.utime(initpy, None)
    except Exception:
        open(initpy, 'wb').close()

os.path.walk(tmpdir, myvisit, None)

for subdir in os.listdir(tmpdir):
    shutil.move(os.path.join(tmpdir, subdir), thisdir)
