const binary = require('node-pre-gyp');
const path = require('path')
// 'lib', 'binding', 'v0.1.1', ['node', 'v' + process.versions.modules, process.platform, process.arch].join('-'), 'deepspeech-bingings.node')
const binding_path = binary.find(path.resolve(path.join(__dirname, 'package.json')));
const binding = require(binding_path);

function Model() {
    this._impl = null;

    const rets = binding.CreateModel.apply(null, arguments);
    const status = rets[0];
    const impl = rets[1];
    if (status !== 0) {
        throw "CreateModel failed with error code " + status;
    }

    this._impl = impl;
}

Model.prototype.enableDecoderWithLM = function() {
    const args = [this._impl].concat(Array.prototype.slice.call(arguments));
    binding.EnableDecoderWithLM.apply(null, args);
}

Model.prototype.stt = function() {
    const args = [this._impl].concat(Array.prototype.slice.call(arguments));
    return binding.SpeechToText.apply(null, args);
}

Model.prototype.setupStream = function() {
    const args = [this._impl].concat(Array.prototype.slice.call(arguments));
    const rets = binding.SetupStream.apply(null, args);
    const status = rets[0];
    const ctx = rets[1];
    if (status !== 0) {
        throw "SetupStream failed with error code " + status;
    }
    return ctx;
}

Model.prototype.feedAudioContent = function() {
    binding.FeedAudioContent.apply(null, arguments);
}

Model.prototype.intermediateDecode = function() {
    binding.IntermediateDecode.apply(null, arguments);
}

Model.prototype.finishStream = function() {
    return binding.FinishStream.apply(null, arguments);
}

module.exports = {
    Model: Model,
    audioToInputVector: binding.AudioToInputVector
};
