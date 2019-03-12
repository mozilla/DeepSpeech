#!/bin/bash

# Give full addresses preferably, and folder paths should end with /

if [ -z $1 ]; then
	echo "Didn't pass data directory"
	exit
fi

if [ -z $2 ]; then
	echo "Didn't pass output embeddings directory"
	exit
fi

if [ -z $3 ]; then
	echo "Didn't pass the Initialize Script executable directory"
	exit
fi
if [ -z $4 ]; then
	echo "Didn't pass the DeepSpeech executable directory"
	exit
fi

$3/initialize_folder.sh $2

echo "wav_filename,transcript" > /tmp/audio_files.csv

for filename in "$1"*; do
    [ -e "$filename" ] || continue
    echo "$filename,noscoretranscript" >> /tmp/audio_files.csv
done

cd $4

source $HOME/tmp/deepspeech-venv/bin/activate
#$4/evaluate.py --test_files /tmp/audio_files.csv --checkpoint_dir $4/checkpoint/ --embeddings_output_dir $2/ --test_batch_size 1
./evaluate.py --test_files /tmp/audio_files.csv --checkpoint_dir checkpoint/ --test_batch_size 1 -embeddings_output_dir $2 --report_count 200
