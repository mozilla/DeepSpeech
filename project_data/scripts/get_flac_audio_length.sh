#!/bin/bash

for filename in /home/ubuntu/data/orig_audio/*.flac; do
    duration=$(soxi -D $filename)
    echo "$filename: $duration"
done