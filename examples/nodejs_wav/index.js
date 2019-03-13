const DeepSpeech = require('deepspeech');
const Fs = require('fs');
const Sox = require('sox-stream');
const MemoryStream = require('memory-stream');
const Duplex = require('stream').Duplex;
const Wav = require('node-wav');

const BEAM_WIDTH = 1024;
const N_FEATURES = 26;
const N_CONTEXT = 9;
let modelPath = './models/output_graph.pbmm';
let alphabetPath = './models/alphabet.txt';

let model = new DeepSpeech.Model(modelPath, N_FEATURES, N_CONTEXT, alphabetPath, BEAM_WIDTH);

const LM_ALPHA = 0.75;
const LM_BETA = 1.85;
let lmPath = './models/lm.binary';
let triePath = './models/trie';

model.enableDecoderWithLM(alphabetPath, lmPath, triePath, LM_ALPHA, LM_BETA);

let audioFile = process.argv[2] || './audio/2830-3980-0043.wav';

if (!Fs.existsSync(audioFile)) {
	console.log('file missing:', audioFile);
	process.exit();
}

const buffer = Fs.readFileSync(audioFile);
const result = Wav.decode(buffer);

if (result.sampleRate < 16000) {
	console.error('Warning: original sample rate (' + result.sampleRate + ') is lower than 16kHz. Up-sampling might produce erratic speech recognition.');
}

function bufferToStream(buffer) {
	let stream = new Duplex();
	stream.push(buffer);
	stream.push(null);
	return stream;
}

let audioStream = new MemoryStream();
bufferToStream(buffer).
pipe(Sox({
	global: {
		'no-dither': true,
	},
	output: {
		bits: 16,
		rate: 16000,
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
	
	const audioLength = (audioBuffer.length / 2) * ( 1 / 16000);
	console.log('audio length', audioLength);
	
	let result = model.stt(audioBuffer.slice(0, audioBuffer.length / 2), 16000);
	
	console.log('result:', result);
});