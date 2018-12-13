.PHONY: clean apk-clean

include ../definitions.mk

LIBDEEPSPEECH_SO ?= ${TFDIR}/bazel-bin/native_client/libdeepspeech.so

all: apk

clean: apk-clean
	rm -rf *.java jni/deepspeech_wrap.cpp

apk-clean:
	./gradlew clean

apk: apk-clean bindings
	LIBDEEPSPEECH_SO=$(LIBDEEPSPEECH_SO) ./gradlew build

bindings: clean
	swig -c++ -java -package deepspeech.mozilla.org.deepspeech -outdir app/src/main/java/deepspeech/mozilla/org/deepspeech/ -o jni/deepspeech_wrap.cpp jni/deepspeech.i 
