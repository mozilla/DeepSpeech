# Microphone VAD Streaming

Stream from microphone to DeepSpeech, using VAD (voice activity detection). A fairly simple example demonstrating the DeepSpeech streaming API in Python. Also useful for quick, real-time testing of models and decoding parameters.

## Installation

```bash
pip install -r requirements.txt
```

Uses portaudio for microphone access, so on Linux, you may need to install its header files to compile the `pyaudio` package:

```bash
sudo apt install portaudio19-dev
```

Installation on MacOS may fail due to portaudio, use brew to install it:

```bash
brew install portaudio
```

## Usage

```
usage: mic_vad_streaming.py [-h] [-v VAD_AGGRESSIVENESS] [--nospinner]
                            [-w SAVEWAV] -m MODEL [-a ALPHABET] [-l LM]
                            [-t TRIE] [-nf N_FEATURES] [-nc N_CONTEXT]
                            [-la LM_ALPHA] [-lb LM_BETA]
                            [-bw BEAM_WIDTH]

Stream from microphone to DeepSpeech using VAD

optional arguments:
  -h, --help            show this help message and exit
  -v VAD_AGGRESSIVENESS, --vad_aggressiveness VAD_AGGRESSIVENESS
                        Set aggressiveness of VAD: an integer between 0 and 3,
                        0 being the least aggressive about filtering out non-
                        speech, 3 the most aggressive. Default: 3
  --nospinner           Disable spinner
  -w SAVEWAV, --savewav SAVEWAV
                        Save .wav files of utterences to given directory
  -m MODEL, --model MODEL
                        Path to the model (protocol buffer binary file, or
                        entire directory containing all standard-named files
                        for model)
  -a ALPHABET, --alphabet ALPHABET
                        Path to the configuration file specifying the alphabet
                        used by the network. Default: alphabet.txt
  -l LM, --lm LM        Path to the language model binary file. Default:
                        lm.binary
  -t TRIE, --trie TRIE  Path to the language model trie file created with
                        native_client/generate_trie. Default: trie
  -nf N_FEATURES, --n_features N_FEATURES
                        Number of MFCC features to use. Default: 26
  -nc N_CONTEXT, --n_context N_CONTEXT
                        Size of the context window used for producing
                        timesteps in the input vector. Default: 9
  -la LM_ALPHA, --lm_alpha LM_ALPHA
                        The alpha hyperparameter of the CTC decoder. Language
                        Model weight. Default: 0.75
  -lb LM_BETA, --lm_beta LM_BETA
                        The beta hyperparameter of the CTC decoder. Word insertion
                        bonus. Default: 1.85
  -bw BEAM_WIDTH, --beam_width BEAM_WIDTH
                        Beam width used in the CTC decoder when building
                        candidate transcriptions. Default: 500
```
