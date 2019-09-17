import sys
import os
import logging
import argparse
import subprocess
import shlex
import numpy as np
import wavTranscriber

# Debug helpers
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main(args):
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--audio', required=False,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--model', required=True,
                        help='Path to directory that contains all model files (output_graph, lm, trie and alphabet)')
    parser.add_argument('--stream', required=False, action='store_true',
                        help='To use deepspeech streaming interface')
    args = parser.parse_args()
    if args.stream is True:
             print("Opening mic for streaming")
    elif args.audio is not None:
            logging.debug("Transcribing audio file @ %s" % args.audio)
    else:
        parser.print_help()
        parser.exit()

    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser(args.model)

    # Resolve all the paths of model files
    output_graph, alphabet, lm, trie = wavTranscriber.resolve_models(dirName)

    # Load output_graph, alpahbet, lm and trie
    model_retval = wavTranscriber.load_model(output_graph, alphabet, lm, trie)

    if args.audio is not None:
        title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'LM Load Time(s)']
        print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

        inference_time = 0.0

        # Run VAD on the input file
        waveFile = args.audio
        segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, args.aggressive)
        f = open(waveFile.rstrip(".wav") + ".txt", 'w')
        logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")

        for i, segment in enumerate(segments):
            # Run deepspeech on the chunk that just completed VAD
            logging.debug("Processing chunk %002d" % (i,))
            audio = np.frombuffer(segment, dtype=np.int16)
            output = wavTranscriber.stt(model_retval[0], audio, sample_rate)
            inference_time += output[1]
            logging.debug("Transcript: %s" % output[0])

            f.write(output[0] + " ")

        # Summary of the files processed
        f.close()

        # Extract filename from the full file path
        filename, ext = os.path.split(os.path.basename(waveFile))
        logging.debug("************************************************************************************************************")
        logging.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
        logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
        logging.debug("************************************************************************************************************")
        print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
    else:
        sctx = model_retval[0].createStream()
        subproc = subprocess.Popen(shlex.split('rec -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'),
                                   stdout=subprocess.PIPE,
                                   bufsize=0)
        print('You can start speaking now. Press Control-C to stop recording.')

        try:
            while True:
                data = subproc.stdout.read(512)
                model_retval[0].feedAudioContent(sctx, np.frombuffer(data, np.int16))
        except KeyboardInterrupt:
            print('Transcription: ', model_retval[0].finishStream(sctx))
            subproc.terminate()
            subproc.wait()


if __name__ == '__main__':
    main(sys.argv[1:])
