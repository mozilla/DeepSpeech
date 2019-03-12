#!/bin/bash

if [ -z $1 ]; then
	echo "Didn't pass any argument"
	exit
fi

if [ ! -d "$1/embeddings/layer4" ]; then
	mkdir -p "$1/embeddings/layer4"
fi
if [ ! -d "$1/embeddings/layer5" ]; then
	mkdir "$1/embeddings/layer5"
fi
if [ ! -d "$1/embeddings/layer6" ]; then
	mkdir "$1/embeddings/layer6"
fi
if [ ! -d "$1/embeddings/text" ]; then
	mkdir "$1/embeddings/text"
fi
