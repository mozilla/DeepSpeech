.PHONY: clean android-clean jre jre-prepare jre-libdeepspeech-jni jre-gradle jre-collect jre-clean android-prepare

include ../definitions.mk

ARCHS := $(shell grep 'ABI_FILTERS' libdeepspeech_android/gradle.properties | cut -d'=' -f2 | sed -e 's/;/ /g')
GRADLE ?= ./gradlew

all: android jre
standalone: jre

clean: android-clean jre-clean
	rm -rf *.java jni/deepspeech_wrap.cpp

android-clean:
	$(GRADLE) clean
	rm -rf settings.gradle

jre-clean:
	rm -f build.gradle
	rm -f jni/deepspeech_wrap.cpp jni/deepspeech_wrap.o
	rm -rf libdeepspeech_jre/cmake_install.cmake libdeepspeech_jre/CMakeCache.txt libdeepspeech_jre/Makefile libdeepspeech_jre/CMakeFiles/
	rm -rf libdeepspeech_jre/build/
	rm -rf settings.gradle

libdeepspeech_jre/libs/%/libdeepspeech.so:
	-mkdir libdeepspeech_jre/libs/$*/
	cp ${TFDIR}/bazel-out/$*-*/bin/native_client/libdeepspeech.so libdeepspeech_jre/libs/$*/

libdeepspeech_android/libs/%/libdeepspeech.so:
	-mkdir libdeepspeech_android/libs/$*/
	cp ${TFDIR}/bazel-out/$*-*/bin/native_client/libdeepspeech.so libdeepspeech_android/libs/$*/

android-prepare:
	cp build.gradle.android build.gradle
	cp libdeepspeech_android/settings.gradle settings.gradle

android: android-prepare android-clean bindings $(patsubst %,libdeepspeech/libs/%/libdeepspeech.so,$(ARCHS))
	$(GRADLE) build

jre: jre-prepare jre-collect jre-restore-makefile jre-clean
jre-prepare: $(patsubst %,libdeepspeech_jre/libs/%/libdeepspeech.so,$(ARCHS))
	cp Makefile Makefile.original
	cp build.gradle.standalone build.gradle
	cp libdeepspeech_jre/settings.gradle ./settings.gradle
	sed -i 's|__JAVA_HOME__|'${JAVA_HOME}'|g' libdeepspeech_jre/CMakeLists.txt

jre-libdeepspeech-jni: bindings
	cd libdeepspeech_jre; \
		cmake .; \
		$(MAKE)

jre-gradle: jre-libdeepspeech-jni
	$(GRADLE) build

jre-collect: jre-gradle
	mkdir -p build
	mv libdeepspeech_jre/libdeepspeech-jni.so build/
	cp libdeepspeech_jre/libs/x86_64/libdeepspeech.so build/
	cp libdeepspeech_jre/build/libs/libdeepspeech_jre.jar build/

jre-restore-makefile:
	mv Makefile.original Makefile

maven-bundle: android
	$(GRADLE) uploadArchives
	$(GRADLE) zipMavenArtifacts

bindings: ds-swig
	mkdir -p ./tmp
	$(DS_SWIG_ENV) swig -c++ -java -package org.deepspeech.libdeepspeech -outdir ./tmp -o jni/deepspeech_wrap.cpp jni/deepspeech.i
	cp ./tmp/* libdeepspeech_android/src/main/java/org/deepspeech/libdeepspeech/
	cp ./tmp/* libdeepspeech_jre/src/main/java/org/deepspeech/libdeepspeech/
	rm -rf ./tmp