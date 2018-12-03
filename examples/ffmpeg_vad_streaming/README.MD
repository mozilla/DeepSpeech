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
node ./index.js --audio <AUDIO_FILE> --model $HOME/models/output_graph.pbmm --alphabet $HOME/models/alphabet.txt
```

Here is an example for a remote RTMP-Stream:
```bash
node ./index.js  --audio rtmp://<IP>:1935/live/teststream --model $HOME/models/output_graph.pbmm --alphabet $HOME/models/alphabet.txt
```
