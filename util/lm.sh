#!/bin/bash


KENLM=$1
NATIVE_CLIENT=$2
TEXT=$3
ALPHABET=$4

# Build pruned LM.
${KENLM}/build/bin/lmplz \
	--order 5 \
       --temp_prefix /tmp/ \
       --memory 50% \
       --text ${TEXT} \
       --arpa ${TEXT}.arpa \
       --prune 0 0 0 1

# Quantize and produce trie binary.
${KENLM}/build/bin/build_binary \
	-a 255 \
        -q 8 \
        trie \
        ${TEXT}.arpa \
        ${TEXT}.arpa.binary


${NATIVE_CLIENT}/generate_trie \
		${ALPHABET} \
		${TEXT}.arpa.binary \
		${TEXT}.trie

echo "PLEASE FIND LM.BINARY AT: /tmp/${TEXT}.arpa.binary"
echo "PLEASE FIND TRIE AT: /tmp/${TEXT}.trie"
