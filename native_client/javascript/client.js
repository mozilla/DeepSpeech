#!/usr/bin/env node
'use strict';

const Fs = require('fs');
const Sox = require('sox-stream');
const Ds = require('./index.js');
const argparse = require('argparse');
const MemoryStream = require('memory-stream');
const Wav = require('node-wav');
const Duplex = require('stream').Duplex;
const util = require('util');

var VersionAction = function VersionAction(options) {
  options = options || {};
  options.nargs = 0;
  argparse.Action.call(this, options);
}
util.inherits(VersionAction, argparse.Action);

VersionAction.prototype.call = function(parser) {
  Ds.printVersions();
  let runtime = 'Node';
  if (process.versions.electron) {
    runtime = 'Electron';
  }
  console.error('Runtime: ' + runtime);
  process.exit(0);
}

var parser = new argparse.ArgumentParser({addHelp: true, description: 'Running DeepSpeech inference.'});
parser.addArgument(['--model'], {required: true, help: 'Path to the model (protocol buffer binary file)'});
parser.addArgument(['--lm'], {help: 'Path to the language model binary file', nargs: '?'});
parser.addArgument(['--trie'], {help: 'Path to the language model trie file created with native_client/generate_trie', nargs: '?'});
parser.addArgument(['--audio'], {required: true, help: 'Path to the audio file to run (WAV format)'});
parser.addArgument(['--beam_width'], {help: 'Beam width for the CTC decoder', defaultValue: 500, type: 'int'});
parser.addArgument(['--lm_alpha'], {help: 'Language model weight (lm_alpha)', defaultValue: 0.75, type: 'float'});
parser.addArgument(['--lm_beta'], {help: 'Word insertion bonus (lm_beta)', defaultValue: 1.85, type: 'float'});
parser.addArgument(['--version'], {action: VersionAction, help: 'Print version and exits'});
parser.addArgument(['--extended'], {action: 'storeTrue', help: 'Output string from extended metadata'});
var args = parser.parseArgs();

function totalTime(hrtimeValue) {
  return (hrtimeValue[0] + hrtimeValue[1] / 1000000000).toPrecision(4);
}

function metadataToString(metadata) {
  var retval = ""
  for (var i = 0; i < metadata.num_items; ++i) {
    retval += metadata.items[i].character;
  }
  Ds.FreeMetadata(metadata);
  return retval;
}

console.error('Loading model from file %s', args['model']);
const model_load_start = process.hrtime();
var model = new Ds.Model(args['model'], args['beam_width']);
const model_load_end = process.hrtime(model_load_start);
console.error('Loaded model in %ds.', totalTime(model_load_end));

var desired_sample_rate = model.sampleRate();

if (args['lm'] && args['trie']) {
  console.error('Loading language model from files %s %s', args['lm'], args['trie']);
  const lm_load_start = process.hrtime();
  model.enableDecoderWithLM(args['lm'], args['trie'], args['lm_alpha'], args['lm_beta']);
  const lm_load_end = process.hrtime(lm_load_start);
  console.error('Loaded language model in %ds.', totalTime(lm_load_end));
}

const buffer = Fs.readFileSync(args['audio']);
const result = Wav.decode(buffer);

if (result.sampleRate < desired_sample_rate) {
  console.error('Warning: original sample rate (' + result.sampleRate + ') ' +
                'is lower than ' + desired_sample_rate + 'Hz. ' +
                'Up-sampling might produce erratic speech recognition.');
}

function bufferToStream(buffer) {
  var stream = new Duplex();
  stream.push(buffer);
  stream.push(null);
  return stream;
}

var audioStream = new MemoryStream();
bufferToStream(buffer).
  pipe(Sox({
    global: {
      'no-dither': true,
    },
    output: {
      bits: 16,
      rate: desired_sample_rate,
      channels: 1,
      encoding: 'signed-integer',
      endian: 'little',
      compression: 0.0,
      type: 'raw'
    }
  })).
  pipe(audioStream);

audioStream.on('finish', () => {
  let audioBuffer = audioStream.toBuffer();

  const inference_start = process.hrtime();
  console.error('Running inference.');
  const audioLength = (audioBuffer.length / 2) * (1 / desired_sample_rate);

  if (args['extended']) {
    console.log(metadataToString(model.sttWithMetadata(audioBuffer)));
  } else {
    console.log(model.stt(audioBuffer));
  }
  const inference_stop = process.hrtime(inference_start);
  console.error('Inference took %ds for %ds audio file.', totalTime(inference_stop), audioLength.toPrecision(4));
  Ds.FreeModel(model);
  process.exit(0);
});
