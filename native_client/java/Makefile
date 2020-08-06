.PHONY: clean apk-clean

include ../definitions.mk

ARCHS := $(shell grep 'ABI_FILTERS' libmozillavoicestt/gradle.properties | cut -d'=' -f2 | sed -e 's/;/ /g')
GRADLE ?= ./gradlew

all: apk

clean: apk-clean
	rm -rf *.java jni/deepspeech_wrap.cpp

apk-clean:
	$(GRADLE) clean

libs-clean:
	rm -fr libmozillavoicestt/libs/*/libmozilla_voice_stt.so

libmozillavoicestt/libs/%/libmozilla_voice_stt.so:
	-mkdir libmozillavoicestt/libs/$*/
	cp ${TFDIR}/bazel-out/$*-*/bin/native_client/libmozilla_voice_stt.so libmozillavoicestt/libs/$*/

apk: apk-clean bindings $(patsubst %,libmozillavoicestt/libs/%/libmozilla_voice_stt.so,$(ARCHS))
	$(GRADLE) build

maven-bundle: apk
	$(GRADLE) uploadArchives
	$(GRADLE) zipMavenArtifacts

bindings: clean ds-swig
	$(DS_SWIG_ENV) swig -c++ -java -package org.mozilla.voice.stt -outdir libmozillavoicestt/src/main/java/org/mozilla/voice/stt/ -o jni/deepspeech_wrap.cpp jni/deepspeech.i
