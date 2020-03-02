'use strict';

const binary = require('node-pre-gyp');
const path = require('path')
// 'lib', 'binding', 'v0.1.1', ['node', 'v' + process.versions.modules, process.platform, process.arch].join('-'), 'deepspeech-bindings.node')
const binding_path = binary.find(path.resolve(path.join(__dirname, 'package.json')));

// On Windows, we can't rely on RPATH being set to $ORIGIN/../ or on
// @loader_path/../ but we can change the PATH to include the proper directory
// for the dynamic linker
if (process.platform === 'win32') {
  const dslib_path = path.resolve(path.join(binding_path, '../..'));
  var oldPath = process.env.PATH;
  process.env['PATH'] = `${dslib_path};${process.env.PATH}`;
}

const binding = require(binding_path);

if (process.platform === 'win32') {
  process.env['PATH'] = oldPath;
}

/**
 * @class
 * An object providing an interface to a trained DeepSpeech model.
 *
 * @param {string} aModelPath The path to the frozen model graph.
 *
 * @throws on error
 */
function Model(aModelPath) {
    this._impl = null;

    const rets = binding.CreateModel(aModelPath);
    const status = rets[0];
    const impl = rets[1];
    if (status !== 0) {
        error_message = binding.ErrorCodeToErrorMessage(status);
        throw "CreateModel failed with error message "+error_message+" with error code 0x" + status.toString(16);
    }

    this._impl = impl;
}

/**
 * Get beam width value used by the model. If :js:func:Model.setBeamWidth was
 * not called before, will return the default value loaded from the model file.
 *
 * @return {number} Beam width value used by the model.
 */
Model.prototype.beamWidth = function() {
    return binding.GetModelBeamWidth(this._impl);
}

/**
 * Set beam width value used by the model.
 *
 * @param {number} The beam width used by the model. A larger beam width value generates better results at the cost of decoding time.
 *
 * @return {number} Zero on success, non-zero on failure.
 */
Model.prototype.setBeamWidth = function(aBeamWidth) {
    return binding.SetModelBeamWidth(this._impl, aBeamWidth);
}

/**
 * Return the sample rate expected by the model.
 *
 * @return {number} Sample rate.
 */
Model.prototype.sampleRate = function() {
    return binding.GetModelSampleRate(this._impl);
}

/**
 * Enable decoding using an external scorer.
 *
 * @param {string} aScorerPath The path to the external scorer file.
 *
 * @return {number} Zero on success, non-zero on failure (invalid arguments).
 */
Model.prototype.enableExternalScorer = function(aScorerPath) {
    return binding.EnableExternalScorer(this._impl, aScorerPath);
}

/**
 * Disable decoding using an external scorer.
 *
 * @return {number} Zero on success, non-zero on failure (invalid arguments).
 */
Model.prototype.disableExternalScorer = function() {
    return binding.EnableExternalScorer(this._impl);
}

/**
 * Set hyperparameters alpha and beta of the external scorer.
 *
 * @param {float} aLMAlpha The alpha hyperparameter of the CTC decoder. Language Model weight.
 * @param {float} aLMBeta The beta hyperparameter of the CTC decoder. Word insertion weight.
 *
 * @return {number} Zero on success, non-zero on failure (invalid arguments).
 */
Model.prototype.setScorerAlphaBeta = function(aLMAlpha, aLMBeta) {
    return binding.SetScorerAlphaBeta(this._impl, aLMAlpha, aLMBeta);
}

/**
 * Use the DeepSpeech model to perform Speech-To-Text.
 *
 * @param {object} aBuffer A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
 *
 * @return {string} The STT result. Returns undefined on error.
 */
Model.prototype.stt = function(aBuffer) {
    return binding.SpeechToText(this._impl, aBuffer);
}

/**
 * Use the DeepSpeech model to perform Speech-To-Text and output metadata
 * about the results.
 *
 * @param {object} aBuffer A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
 *
 * @return {object} Outputs a :js:func:`Metadata` struct of individual letters along with their timing information. The user is responsible for freeing Metadata by calling :js:func:`FreeMetadata`. Returns undefined on error.
 */
Model.prototype.sttWithMetadata = function(aBuffer) {
    return binding.SpeechToTextWithMetadata(this._impl, aBuffer);
}

/**
 * Create a new streaming inference state. One can then call :js:func:`Stream.feedAudioContent` and :js:func:`Stream.finishStream` on the returned stream object.
 *
 * @return {object} a :js:func:`Stream` object that represents the streaming state.
 *
 * @throws on error
 */
Model.prototype.createStream = function() {
    const rets = binding.CreateStream(this._impl);
    const status = rets[0];
    const ctx = rets[1];
    if (status !== 0) {
        error_message = binding.ErrorCodeToErrorMessage(status);
        throw "CreateStream failed with error message "+error_message+" with error code 0x" + status.toString(16);
    }
    return ctx;
}

/**
 * @class
 * Provides an interface to a DeepSpeech stream. The constructor cannot be called
 * directly, use :js:func:`Model.createStream`.
 */
function Stream(nativeStream) {
    this._impl = nativeStream;
}

/**
 * Feed audio samples to an ongoing streaming inference.
 *
 * @param {buffer} aBuffer An array of 16-bit, mono raw audio samples at the
 *                 appropriate sample rate (matching what the model was trained on).
 */
Stream.prototype.feedAudioContent = function(aBuffer) {
    binding.FeedAudioContent(this._impl, aBuffer);
}

/**
 * Compute the intermediate decoding of an ongoing streaming inference.
 *
 * @return {string} The STT intermediate result.
 */
Stream.prototype.intermediateDecode = function() {
    return binding.IntermediateDecode(this._impl);
}

/**
 * Signal the end of an audio signal to an ongoing streaming inference, returns the STT result over the whole audio signal.
 *
 * @return {string} The STT result.
 *
 * This method will free the stream, it must not be used after this method is called.
 */
Stream.prototype.finishStream = function() {
    result = binding.FinishStream(this._impl);
    this._impl = null;
    return result;
}

/**
 * Signal the end of an audio signal to an ongoing streaming inference, returns per-letter metadata.
 *
 * @return {object} Outputs a :js:func:`Metadata` struct of individual letters along with their timing information. The user is responsible for freeing Metadata by calling :js:func:`FreeMetadata`.
 *
 * This method will free the stream, it must not be used after this method is called.
 */
Stream.prototype.finishStreamWithMetadata = function() {
    result = binding.FinishStreamWithMetadata(this._impl);
    this._impl = null;
    return result;
}


/**
 * Frees associated resources and destroys model object.
 *
 * @param {object} model A model pointer returned by :js:func:`Model`
 *
 */
function FreeModel(model) {
    return binding.FreeModel(model._impl);
}

/**
 * Free memory allocated for metadata information.
 *
 * @param {object} metadata Object containing metadata as returned by :js:func:`Model.sttWithMetadata` or :js:func:`Model.finishStreamWithMetadata`
 */
function FreeMetadata(metadata) {
    return binding.FreeMetadata(metadata);
}

/**
 * Destroy a streaming state without decoding the computed logits. This
 * can be used if you no longer need the result of an ongoing streaming
 * inference and don't want to perform a costly decode operation.
 *
 * @param {Object} stream A stream object returned by :js:func:`Model.createStream`.
 */
function FreeStream(stream) {
    return binding.FreeStream(stream._impl);
}

/**
 * Print version of this library and of the linked TensorFlow library on standard output.
 */
function Version() {
    return binding.Version();
}


//// Metadata and MetadataItem are here only for documentation purposes

/**
 * @class
 * 
 * Stores each individual character, along with its timing information
 */
function MetadataItem() {}

/** 
 * The character generated for transcription
 *
 * @return {string} The character generated
 */
MetadataItem.prototype.character = function() {}

/**
 * Position of the character in units of 20ms
 *
 * @return {int} The position of the character
 */
MetadataItem.prototype.timestep = function() {};

/**
 * Position of the character in seconds
 *
 * @return {float} The position of the character
 */
MetadataItem.prototype.start_time = function() {};

/**
 * @class
 *
 * Stores the entire CTC output as an array of character metadata objects
 */
function Metadata () {}

/**
 * List of items
 *
 * @return {array} List of :js:func:`MetadataItem`
 */
Metadata.prototype.items = function() {}

/**
 * Size of the list of items
 *
 * @return {int} Number of items
 */
Metadata.prototype.num_items = function() {}

/**
 * Approximated confidence value for this transcription. This is roughly the
 * sum of the acoustic model logit values for each timestep/character that
 * contributed to the creation of this transcription.
 *
 * @return {float} Confidence value
 */
Metadata.prototype.confidence = function() {}

module.exports = {
    Model: Model,
    Metadata: Metadata,
    MetadataItem: MetadataItem,
    Version: Version,
    FreeModel: FreeModel,
    FreeStream: FreeStream,
    FreeMetadata: FreeMetadata
};
