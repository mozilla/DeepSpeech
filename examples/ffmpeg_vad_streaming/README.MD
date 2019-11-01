# FFmpeg VAD Streaming

Streaming inference from arbitrary source (FFmpeg input) to DeepSpeech, using VAD (voice activity detection). A fairly simple example demonstrating the DeepSpeech streaming API in Node.js.

This example was successfully tested with a mobile phone streaming a live feed to a RTMP server (nginx-rtmp), which then could be used by this script for near real time speech recognition.

## Installation

```bash
npm install
```

Moreover FFmpeg must be installed:

```bash
sudo apt-get install ffmpeg
```

## Usage

Here is an example for a local audio file:
```bash
node ./index.js --audio <AUDIO_FILE> \
                --model $HOME/models/output_graph.pbmm \
```

Here is an example for a remote RTMP-Stream:
```bash
node ./index.js  --audio rtmp://<IP>:1935/live/teststream \
                 --model $HOME/models/output_graph.pbmm \
```

## Examples
Real time streaming inference with DeepSpeech's example audio ([audio-0.4.1.tar.gz](https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/audio-0.4.1.tar.gz)).
```bash
node ./index.js --audio $HOME/audio/2830-3980-0043.wav \
                --lm $HOME/models/lm.binary \
                --trie $HOME/models/trie \
                --model $HOME/models/output_graph.pbmm \
```
```bash
node ./index.js --audio $HOME/audio/4507-16021-0012.wav \
                --lm $HOME/models/lm.binary \
                --trie $HOME/models/trie \
                --model $HOME/models/output_graph.pbmm \
```
```bash
node ./index.js --audio $HOME/audio/8455-210777-0068.wav \
                --lm $HOME/models/lm.binary \
                --trie $HOME/models/trie \
                --model $HOME/models/output_graph.pbmm \
```
Real time streaming inference in combination with a RTMP server.
```bash
node ./index.js --audio rtmp://<HOST>/<APP>/<KEY> \
                --lm $HOME/models/lm.binary \
                --trie $HOME/models/trie \
                --model $HOME/models/output_graph.pbmm \
```

## Notes
To get the best result mapped on to your own scenario, it might be helpful to adjust the parameters `VAD_MODE` and `DEBUNCE_TIME`.