#!/bin/sh

if [ -z $1 ]; then
        echo "Didn't pass any argument"
        exit
fi
if [ -d "$1/embeddings/layer4" ]; then
	rm "$1/embeddings/layer4"/*
fi
if [ -d "$1/embeddings/layer5" ]; then
	rm "$1/embeddings/layer5"/*
fi
if [ -d "$1/embeddings/layer6" ]; then
	rm "$1/embeddings/layer6"/*
fi
if [ -d "$1/embeddings/text" ]; then
	rm "$1/embeddings/text"/*
fi
