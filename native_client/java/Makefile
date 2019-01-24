.PHONY: clean apk-clean

include ../definitions.mk

ARCHS := $(shell grep 'ABI_FILTERS' libdeepspeech/gradle.properties | cut -d'=' -f2 | sed -e 's/;/ /g')

all: apk

clean: apk-clean
	rm -rf *.java jni/deepspeech_wrap.cpp

apk-clean:
	./gradlew clean

libs-clean:
	rm -fr libdeepspeech/libs/*/libdeepspeech.so

libdeepspeech/libs/%/libdeepspeech.so:
	-mkdir libdeepspeech/libs/$*/
	cp ${TFDIR}/bazel-out/$*-*/bin/native_client/libdeepspeech.so libdeepspeech/libs/$*/

apk: apk-clean bindings $(patsubst %,libdeepspeech/libs/%/libdeepspeech.so,$(ARCHS))
	./gradlew build
	./gradlew writeNewPom

bindings: clean
	swig -c++ -java -package org.mozilla.deepspeech.libdeepspeech -outdir libdeepspeech/src/main/java/org/mozilla/deepspeech/libdeepspeech/ -o jni/deepspeech_wrap.cpp jni/deepspeech.i
