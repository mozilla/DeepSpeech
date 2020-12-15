#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import absl.app
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import shutil

from .deepspeech_model import create_inference_graph
from .util.checkpoints import load_graph_for_evaluation
from .util.config import Config, initialize_globals
from .util.flags import create_flags, FLAGS
from .util.io import open_remote, rmtree_remote, listdir_remote, is_remote_path, isdir_remote
from .util.logging import log_error, log_info


def file_relative_read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')

    inputs, outputs, _ = create_inference_graph(batch_size=FLAGS.export_batch_size, n_steps=FLAGS.n_steps, tflite=FLAGS.export_tflite)

    graph_version = int(file_relative_read('GRAPH_VERSION').strip())
    assert graph_version > 0

    outputs['metadata_version'] = tf.constant([graph_version], name='metadata_version')
    outputs['metadata_sample_rate'] = tf.constant([FLAGS.audio_sample_rate], name='metadata_sample_rate')
    outputs['metadata_feature_win_len'] = tf.constant([FLAGS.feature_win_len], name='metadata_feature_win_len')
    outputs['metadata_feature_win_step'] = tf.constant([FLAGS.feature_win_step], name='metadata_feature_win_step')
    outputs['metadata_beam_width'] = tf.constant([FLAGS.export_beam_width], name='metadata_beam_width')
    outputs['metadata_alphabet'] = tf.constant([Config.alphabet.Serialize()], name='metadata_alphabet')

    if FLAGS.export_language:
        outputs['metadata_language'] = tf.constant([FLAGS.export_language.encode('utf-8')], name='metadata_language')

    # Prevent further graph changes
    tfv1.get_default_graph().finalize()

    output_names_tensors = [tensor.op.name for tensor in outputs.values() if isinstance(tensor, tf.Tensor)]
    output_names_ops = [op.name for op in outputs.values() if isinstance(op, tf.Operation)]
    output_names = output_names_tensors + output_names_ops

    with tf.Session() as session:
        # Restore variables from checkpoint
        load_graph_for_evaluation(session)

        output_filename = FLAGS.export_file_name + '.pb'
        if FLAGS.remove_export:
            if isdir_remote(FLAGS.export_dir):
                log_info('Removing old export')
                rmtree_remote(FLAGS.export_dir)

        output_graph_path = os.path.join(FLAGS.export_dir, output_filename)

        if not is_remote_path(FLAGS.export_dir) and not os.path.isdir(FLAGS.export_dir):
            os.makedirs(FLAGS.export_dir)

        frozen_graph = tfv1.graph_util.convert_variables_to_constants(
            sess=session,
            input_graph_def=tfv1.get_default_graph().as_graph_def(),
            output_node_names=output_names)

        frozen_graph = tfv1.graph_util.extract_sub_graph(
            graph_def=frozen_graph,
            dest_nodes=output_names)

        if not FLAGS.export_tflite:
            with open_remote(output_graph_path, 'wb') as fout:
                fout.write(frozen_graph.SerializeToString())
        else:
            output_tflite_path = os.path.join(FLAGS.export_dir, output_filename.replace('.pb', '.tflite'))

            converter = tf.lite.TFLiteConverter(frozen_graph, input_tensors=inputs.values(), output_tensors=outputs.values())
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # AudioSpectrogram and Mfcc ops are custom but have built-in kernels in TFLite
            converter.allow_custom_ops = True
            tflite_model = converter.convert()

            with open_remote(output_tflite_path, 'wb') as fout:
                fout.write(tflite_model)

        log_info('Models exported at %s' % (FLAGS.export_dir))

    metadata_fname = os.path.join(FLAGS.export_dir, '{}_{}_{}.md'.format(
        FLAGS.export_author_id,
        FLAGS.export_model_name,
        FLAGS.export_model_version))

    model_runtime = 'tflite' if FLAGS.export_tflite else 'tensorflow'
    with open_remote(metadata_fname, 'w') as f:
        f.write('---\n')
        f.write('author: {}\n'.format(FLAGS.export_author_id))
        f.write('model_name: {}\n'.format(FLAGS.export_model_name))
        f.write('model_version: {}\n'.format(FLAGS.export_model_version))
        f.write('contact_info: {}\n'.format(FLAGS.export_contact_info))
        f.write('license: {}\n'.format(FLAGS.export_license))
        f.write('language: {}\n'.format(FLAGS.export_language))
        f.write('runtime: {}\n'.format(model_runtime))
        f.write('min_ds_version: {}\n'.format(FLAGS.export_min_ds_version))
        f.write('max_ds_version: {}\n'.format(FLAGS.export_max_ds_version))
        f.write('acoustic_model_url: <replace this with a publicly available URL of the acoustic model>\n')
        f.write('scorer_url: <replace this with a publicly available URL of the scorer, if present>\n')
        f.write('---\n')
        f.write('{}\n'.format(FLAGS.export_description))

    log_info('Model metadata file saved to {}. Before submitting the exported model for publishing make sure all information in the metadata file is correct, and complete the URL fields.'.format(metadata_fname))


def package_zip():
    # --export_dir path/to/export/LANG_CODE/ => path/to/export/LANG_CODE.zip
    export_dir = os.path.join(os.path.abspath(FLAGS.export_dir), '') # Force ending '/'
    if is_remote_path(export_dir):
        log_error("Cannot package remote path zip %s. Please do this manually." % export_dir)
        return

    zip_filename = os.path.dirname(export_dir)

    shutil.copy(FLAGS.scorer_path, export_dir)

    archive = shutil.make_archive(zip_filename, 'zip', export_dir)
    log_info('Exported packaged model {}'.format(archive))


def main(_):
    initialize_globals()

    if FLAGS.export_dir:
        tfv1.reset_default_graph()

        if not FLAGS.export_zip:
            # Export to folder
            export()
        else:
            # Export and zip, TFLite only, creates package readable by Java example app
            FLAGS.export_tflite = True

            if listdir_remote(FLAGS.export_dir):
                log_error('Directory {} is not empty, please fix this.'.format(FLAGS.export_dir))
                sys.exit(1)

            export()
            package_zip()
    else:
        log_error('Calling export script directly but no --export_dir specified')
        sys.exit(1)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
