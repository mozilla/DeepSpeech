#ifndef _Included_org_mozilla_deepspeech_NativeImpl
#define _Included_org_mozilla_deepspeech_NativeImpl

#include <jni.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <deepspeech.h>

#ifdef __cplusplus
extern "C" {
#endif

enum BuildConfiguration {
    INVALID = -1, CPU = 0, CUDA = 1
};

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_DeepSpeech_nCreateModel
        (JNIEnv *, jclass, jstring, jlong, jlong, jstring, jlong, jobject);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_DeepSpeech_destroyModel
        (JNIEnv *, jclass, jlong);

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_DeepSpeech_enableDecoderWithLM
        (JNIEnv *, jclass, jlong, jstring, jstring, jstring, jfloat, jfloat);

JNIEXPORT jstring JNICALL Java_org_mozilla_deepspeech_DeepSpeech_nSpeechToText
        (JNIEnv *, jclass, jlong, jobject, jlong, jlong);

JNIEXPORT jstring JNICALL Java_org_mozilla_deepspeech_DeepSpeech_speechToTextUnsafe
        (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

JNIEXPORT jlong JNICALL Java_org_mozilla_deepspeech_DeepSpeech_nSpeechToTextWithMetadata
        (JNIEnv *, jclass, jlong, jobject, jlong, jlong);

JNIEXPORT jlong JNICALL Java_org_mozilla_deepspeech_DeepSpeech_speechToTextWithMetadataUnsafe
        (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_DeepSpeech_nSetupStream
        (JNIEnv *, jclass, jlong, jlong, jlong, jobject);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_DeepSpeech_nFeedAudioContent
        (JNIEnv *, jclass, jlong, jobject, jlong);

JNIEXPORT jstring JNICALL Java_org_mozilla_deepspeech_DeepSpeech_intermediateDecode
        (JNIEnv *, jclass, jlong);

JNIEXPORT jstring JNICALL Java_org_mozilla_deepspeech_DeepSpeech_finishStream
        (JNIEnv *, jclass, jlong);

JNIEXPORT jlong JNICALL Java_org_mozilla_deepspeech_DeepSpeech_finishStreamWithMetadata
        (JNIEnv *, jclass, jlong);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_DeepSpeech_discardStream
        (JNIEnv *, jclass, jlong);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_DeepSpeech_freeMetadata
        (JNIEnv *, jclass, jlong);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_DeepSpeech_printVersions
        (JNIEnv *, jclass);

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_DeepSpeech_nGetConfiguration
        (JNIEnv *, jclass);

#ifdef __cplusplus
}
#endif
#endif
