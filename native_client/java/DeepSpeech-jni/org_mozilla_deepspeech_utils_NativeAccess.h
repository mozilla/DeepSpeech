#include <jni.h>
#include <stdlib.h>
#include <string.h>

#ifndef _Included_org_mozilla_deepspeech_utils_NativeAccess
#define _Included_org_mozilla_deepspeech_utils_NativeAccess

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jstring JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeCString
        (JNIEnv *, jclass, jlong);

JNIEXPORT jchar JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeChar
        (JNIEnv *, jclass, jlong);

JNIEXPORT jshort JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeShort
        (JNIEnv *, jclass, jlong);

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeInt
        (JNIEnv *, jclass, jlong);

JNIEXPORT jlong JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeLong
        (JNIEnv *, jclass, jlong);

JNIEXPORT jlong JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativePointer
        (JNIEnv *, jclass, jlong);

JNIEXPORT jdouble JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeDouble
        (JNIEnv *, jclass, jlong);

JNIEXPORT jfloat JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeFloat
        (JNIEnv *, jclass, jlong);

JNIEXPORT jboolean JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeBoolean
        (JNIEnv *, jclass, jlong);

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeIntSize
        (JNIEnv *, jclass);

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_getNativePointerSize
        (JNIEnv *, jclass);

JNIEXPORT jlong JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_allocateMemory
        (JNIEnv *, jclass, jlong);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_freeMemory
        (JNIEnv *, jclass, jlong);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_copyMemory
        (JNIEnv *, jclass, jlong, jlong, jlong);

JNIEXPORT void JNICALL Java_org_mozilla_deepspeech_utils_NativeAccess_writeByte
        (JNIEnv *, jclass, jbyte, jlong);

#ifdef __cplusplus
}
#endif
#endif
