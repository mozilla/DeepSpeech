#!/usr/bin/env node

const VAD = require("node-vad");
const Ds = require('deepspeech');
const argparse = require('argparse');
const util = require('util');

// These constants control the beam search decoder

// Beam width used in the CTC decoder when building candidate transcriptions
const BEAM_WIDTH = 1024;

// The alpha hyperparameter of the CTC decoder. Language Model weight
const LM_ALPHA = 0.75;

// The beta hyperparameter of the CTC decoder. Word insertion bonus.
const LM_BETA = 1.85;

// These constants are tied to the shape of the graph used (changing them changes
// the geometry of the first layer), so make sure you use the same constants that
// were used during training

// Number of MFCC features to use
const N_FEATURES = 26;

// Size of the context window used for producing timesteps in the input vector
const N_CONTEXT = 9;

let VersionAction = function VersionAction(options) {
	options = options || {};
	options.nargs = 0;
	argparse.Action.call(this, options);
};

util.inherits(VersionAction, argparse.Action);

VersionAction.prototype.call = function(parser) {
	Ds.printVersions();
	process.exit(0);
};

let parser = new argparse.ArgumentParser({addHelp: true, description: 'Running DeepSpeech inference.'});
parser.addArgument(['--model'], {required: true, help: 'Path to the model (protocol buffer binary file)'});
parser.addArgument(['--alphabet'], {required: true, help: 'Path to the configuration file specifying the alphabet used by the network'});
parser.addArgument(['--lm'], {help: 'Path to the language model binary file', nargs: '?'});
parser.addArgument(['--trie'], {help: 'Path to the language model trie file created with native_client/generate_trie', nargs: '?'});
parser.addArgument(['--audio'], {required: true, help: 'Path to the audio file to run (WAV format)'});
parser.addArgument(['--version'], {action: VersionAction, help: 'Print version and exits'});
let args = parser.parseArgs();

function totalTime(hrtimeValue) {
	return (hrtimeValue[0] + hrtimeValue[1] / 1000000000).toPrecision(4);
}

console.error('Loading model from file %s', args['model']);
const model_load_start = process.hrtime();
let model = new Ds.Model(args['model'], N_FEATURES, N_CONTEXT, args['alphabet'], BEAM_WIDTH);
const model_load_end = process.hrtime(model_load_start);
console.error('Loaded model in %ds.', totalTime(model_load_end));

if (args['lm'] && args['trie']) {
	console.error('Loading language model from files %s %s', args['lm'], args['trie']);
	const lm_load_start = process.hrtime();
	model.enableDecoderWithLM(args['alphabet'], args['lm'], args['trie'],
		LM_ALPHA, LM_BETA);
	const lm_load_end = process.hrtime(lm_load_start);
	console.error('Loaded language model in %ds.', totalTime(lm_load_end));
}

const vad = new VAD(VAD.Mode.NORMAL);
const voice = {START: true, STOP: false};
let sctx = model.setupStream(150, 16000);
let state = voice.STOP;

function finishStream() {
	const model_load_start = process.hrtime();
	console.error('Running inference.');
	console.log('Transcription: ', model.finishStream(sctx));
	const model_load_end = process.hrtime(model_load_start);
	console.error('Inference took %ds.', totalTime(model_load_end));
}

let ffmpeg = require('child_process').spawn('ffmpeg', [
	'-hide_banner',
	'-nostats',
	'-loglevel', 'fatal',
	'-i', args['audio'],
	'-af', 'highpass=f=200,lowpass=f=3000',
	'-vn',
	'-acodec', 'pcm_s16le',
	'-ac', 1,
	'-ar', 16000,
	'-f', 's16le',
	'pipe:'
]);

ffmpeg.stdout.on('data', chunk => {
	vad.processAudio(chunk, 16000).then(res => {
		switch (res) {
			case VAD.Event.SILENCE:
				if (state === voice.START) {
					state = voice.STOP;
					finishStream();
					sctx = model.setupStream(150,16000);
				}
				break;
			case VAD.Event.VOICE:
				state = voice.START;
				model.feedAudioContent(sctx, chunk.slice(0, chunk.length / 2));
				break;
		}
	});
});

ffmpeg.stdout.on('close', code => {
	finishStream();
});
