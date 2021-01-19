.PHONY: clean apk-clean jre jre-prepare jre-libdeepspeech-jni jre-gradle jre-collect jre-clean apk-prepare

include ../definitions.mk

ARCHS := $(shell grep 'ABI_FILTERS' libdeepspeech/gradle.properties | cut -d'=' -f2 | sed -e 's/;/ /g')
GRADLE ?= ./gradlew

all: apk jre
android: apk
standalone: jre

clean: apk-clean jre-clean
	rm -rf *.java jni/deepspeech_wrap.cpp

apk-clean:
	$(GRADLE) clean

libs-clean:
	rm -fr libdeepspeech/libs/*/libdeepspeech.so

libdeepspeech/libs/%/libdeepspeech.so:
	-mkdir libdeepspeech/libs/$*/
	cp ${TFDIR}/bazel-out/$*-*/bin/native_client/libdeepspeech.so libdeepspeech/libs/$*/

apk-prepare:
	cp build.gradle.android build.gradle
	cp libdeepspeech/build.gradle.android libdeepspeech/build.gradle
	cp libdeepspeech/CMakeLists_android.txt libdeepspeech/CMakeLists.txt

apk: apk-prepare apk-clean bindings $(patsubst %,libdeepspeech/libs/%/libdeepspeech.so,$(ARCHS))
	$(GRADLE) build

jre: jre-prepare jre-collect jre-restore-makefile jre-clean
jre-prepare: $(patsubst %,libdeepspeech/libs/%/libdeepspeech.so,$(ARCHS))
	cp Makefile Makefile.original
	cp build.gradle.standalone build.gradle
	cp libdeepspeech/build.gradle.standalone libdeepspeech/build.gradle
	cp libdeepspeech/CMakeLists_standalone.txt libdeepspeech/CMakeLists.txt
	sed -i 's|__JAVA_HOME__|'${JAVA_HOME}'|g' libdeepspeech/CMakeLists.txt

jre-libdeepspeech-jni: bindings
	cd libdeepspeech; \
		cmake .; \
		$(MAKE)

jre-gradle: jre-libdeepspeech-jni
	$(GRADLE) build

jre-collect: jre-gradle
	mkdir -p build
	mv libdeepspeech/libdeepspeech-jni.so build/
	cp libdeepspeech/libs/x86_64/libdeepspeech.so build/
	cp libdeepspeech/build/libs/libdeepspeech.jar build/

jre-clean:
	rm -f libdeepspeech/CMakeLists.txt
	rm -f libdeepspeech/build.gradle
	rm -f build.gradle
	rm -f jni/deepspeech_wrap.cpp jni/deepspeech_wrap.o
	rm -rf libdeepspeech/cmake_install.cmake libdeepspeech/CMakeCache.txt libdeepspeech/Makefile libdeepspeech/CMakeFiles/
	rm -rf libdeepspeech/build/

jre-restore-makefile:
	mv Makefile.original Makefile

maven-bundle: apk
	$(GRADLE) uploadArchives
	$(GRADLE) zipMavenArtifacts

bindings: clean ds-swig
	$(DS_SWIG_ENV) swig -c++ -java -package org.deepspeech.libdeepspeech -outdir libdeepspeech/src/main/java/org/deepspeech/libdeepspeech/ -o jni/deepspeech_wrap.cpp jni/deepspeech.i