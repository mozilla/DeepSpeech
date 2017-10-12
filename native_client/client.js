const Fs = require('fs');
const Sox = require('sox-stream');
const Ds = require('deepspeech');
const MemoryStream = require('memory-stream');

const BEAM_WIDTH = 500;
const LM_WEIGHT = 2.15;
const WORD_COUNT_WEIGHT = -0.10;
const VALID_WORD_COUNT_WEIGHT = 1.10;

const N_FEATURES = 26;
const N_CONTEXT = 9;

var audioStream = new MemoryStream();
Fs.createReadStream(process.argv[3]).
  pipe(Sox({ output: { bits: 16, rate: 16000, channels: 1, type: 'raw' } })).
  pipe(audioStream);
audioStream.on('finish', () => {
  audioBuffer = audioStream.toBuffer();
  var model = new Ds.Model(process.argv[2], N_FEATURES, N_CONTEXT, process.argv[4], BEAM_WIDTH);

  if (process.argv.length > 6) {
     model.enableDecoderWithLM(process.argv[4], process.argv[5], process.argv[6],
                               LM_WEIGHT, WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT);
  }
  // We take half of the buffer_size because buffer is a char* while
  // LocalDsSTT() expected a short*
  console.log(model.stt(audioBuffer.slice(0, audioBuffer.length / 2), 16000));
});
