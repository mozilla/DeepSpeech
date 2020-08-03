.PHONY: clean apk-clean

include ../definitions.mk

ARCHS := $(shell grep 'ABI_FILTERS' libdeepspeech/gradle.properties | cut -d'=' -f2 | sed -e 's/;/ /g')
GRADLE ?= ./gradlew

all: apk

clean: apk-clean
	rm -rf *.java jni/deepspeech_wrap.cpp

apk-clean:
	$(GRADLE) clean

libs-clean:
	rm -fr libdeepspeech/libs/*/libmozilla_voice_stt.so

libdeepspeech/libs/%/libmozilla_voice_stt.so:
	-mkdir libdeepspeech/libs/$*/
	cp ${TFDIR}/bazel-out/$*-*/bin/native_client/libmozilla_voice_stt.so libdeepspeech/libs/$*/

apk: apk-clean bindings $(patsubst %,libdeepspeech/libs/%/libmozilla_voice_stt.so,$(ARCHS))
	$(GRADLE) build

maven-bundle: apk
	$(GRADLE) uploadArchives
	$(GRADLE) zipMavenArtifacts

bindings: clean ds-swig
	$(DS_SWIG_ENV) swig -c++ -java -package org.mozilla.deepspeech.libdeepspeech -outdir libdeepspeech/src/main/java/org/mozilla/deepspeech/libdeepspeech/ -o jni/deepspeech_wrap.cpp jni/deepspeech.i
