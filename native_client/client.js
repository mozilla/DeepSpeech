const Fs = require('fs');
const Sox = require('sox-stream');
const Ds = require('deepspeech');
const ArgumentParser = require('argparse').ArgumentParser;
const MemoryStream = require('memory-stream');

// These constants control the beam search decoder

// Beam width used in the CTC decoder when building candidate transcriptions
const BEAM_WIDTH = 500;

// The alpha hyperparameter of the CTC decoder. Language Model weight
const LM_WEIGHT = 2.15;

// The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
const WORD_COUNT_WEIGHT = -0.10;

// Valid word insertion weight. This is used to lessen the word insertion penalty
// when the inserted word is part of the vocabulary
const VALID_WORD_COUNT_WEIGHT = 1.10;


// These constants are tied to the shape of the graph used (changing them changes
// the geometry of the first layer), so make sure you use the same constants that
// were used during training

// Number of MFCC features to use
const N_FEATURES = 26;

// Size of the context window used for producing timesteps in the input vector
const N_CONTEXT = 9;

var parser = new ArgumentParser({addHelp: true});
parser.addArgument(['model'], {help: 'Path to the model (protocol buffer binary file)'});
parser.addArgument(['audio'], {help: 'Path to the audio file to run (WAV format)'});
parser.addArgument(['alphabet'], {help: 'Path to the configuration file specifying the alphabet used by the network'});
parser.addArgument(['lm'], {help: 'Path to the language model binary file', nargs: '?'});
parser.addArgument(['trie'], {help: 'Path to the language model trie file created with native_client/generate_trie', nargs: '?'});
var args = parser.parseArgs();

var audioStream = new MemoryStream();
Fs.createReadStream(args['audio']).
  pipe(Sox({ output: { bits: 16, rate: 16000, channels: 1, type: 'raw' } })).
  pipe(audioStream);
audioStream.on('finish', () => {
  audioBuffer = audioStream.toBuffer();
  var model = new Ds.Model(args['model'], N_FEATURES, N_CONTEXT, args['alphabet'], BEAM_WIDTH);

  if (args['lm'] && args['trie']) {
    model.enableDecoderWithLM(args['alphabet'], args['lm'], args['trie'],
                              LM_WEIGHT, WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT);
  }
  // We take half of the buffer_size because buffer is a char* while
  // LocalDsSTT() expected a short*
  console.log(model.stt(audioBuffer.slice(0, audioBuffer.length / 2), 16000));
});
