const Fs = require('fs');
const Sox = require('sox-stream');
const Ds = require('deepspeech');
const MemoryStream = require('memory-stream');

var audioStream = new MemoryStream();
Fs.createReadStream(process.argv[3]).
  pipe(Sox({ output: { bits: 16, rate: 16000, channels: 1, type: 'raw' } })).
  pipe(audioStream);
audioStream.on('finish', () => {
  audioBuffer = audioStream.toBuffer();
  var model = new Ds.Model(process.argv[2], 26, 9);
  // We take half of the buffer_size because buffer is a char* while
  // LocalDsSTT() expected a short*
  console.log(model.stt(audioBuffer.slice(0, audioBuffer.length / 2), 16000));
});
