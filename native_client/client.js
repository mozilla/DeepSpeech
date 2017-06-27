const Fs = require('fs');
const Sox = require('sox-stream');
const Ds = require('./javascript/build/Release/deepspeech');
const MemoryStream = require('memory-stream');

var audioStream = new MemoryStream();
Fs.createReadStream(process.argv[3]).
  pipe(Sox({ output: { bits: 16, rate: 16000, channels: 1, type: 'raw' } })).
  pipe(audioStream);
audioStream.on('finish', () => {
  audioBuffer = audioStream.toBuffer();
  var model = new Ds.Model(process.argv[2], 26, 9);
  console.log(model.stt(audioBuffer, 16000));
});
