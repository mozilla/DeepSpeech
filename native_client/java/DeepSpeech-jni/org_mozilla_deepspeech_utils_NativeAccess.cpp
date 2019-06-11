#include "org_mozilla_deepspeech_utils_NativeAccess.h"

jstring Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeCString(JNIEnv *env, jclass, jlong strptr) {
    return env->NewStringUTF((char *) strptr);
}

jchar Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeChar(JNIEnv *, jclass, jlong charPtr) {
    return static_cast<jchar>(*(char *) charPtr);
}

jboolean Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeBoolean(JNIEnv *, jclass, jlong booleanPointer) {
    return static_cast<jboolean>((*(int *) booleanPointer) != 0);
}

jshort Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeShort(JNIEnv *, jclass, jlong shortPtr) {
    return static_cast<jshort>(*(short *) shortPtr);
}

jint Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeInt(JNIEnv *, jclass, jlong intPtr) {
    return static_cast<jint>(*(int *) intPtr);
}

jlong Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeLong(JNIEnv *, jclass, jlong longPtr) {
    return static_cast<jlong>(*(long *) longPtr);
}

jlong Java_org_mozilla_deepspeech_utils_NativeAccess_getNativePointer(JNIEnv *, jclass, jlong pointerPointer) {
    void **ptr = (void **) pointerPointer;
    return (jlong) *ptr;
}

jdouble Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeDouble(JNIEnv *, jclass, jlong doublePtr) {
    return static_cast<jdouble>(*(double *) doublePtr);
}

jfloat Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeFloat(JNIEnv *, jclass, jlong floatPtr) {
    return static_cast<jfloat>(*(float *) floatPtr);
}

jint Java_org_mozilla_deepspeech_utils_NativeAccess_getNativePointerSize(JNIEnv *, jclass) {
    return sizeof(void *);
}

jint Java_org_mozilla_deepspeech_utils_NativeAccess_getNativeIntSize(JNIEnv *, jclass) {
    return sizeof(int);
}

jlong Java_org_mozilla_deepspeech_utils_NativeAccess_allocateMemory(JNIEnv *, jclass, jlong bytes) {
    return (jlong) calloc((size_t) bytes, 1);
}

void Java_org_mozilla_deepspeech_utils_NativeAccess_freeMemory(JNIEnv *, jclass, jlong memory) {
    free((void *) memory);
}

void Java_org_mozilla_deepspeech_utils_NativeAccess_copyMemory(JNIEnv *, jclass, jlong dst, jlong src, jlong numBytes) {
    memcpy((void *) dst, (void *) src, (size_t) numBytes);
}

void Java_org_mozilla_deepspeech_utils_NativeAccess_writeByte(JNIEnv *, jclass, jbyte byte, jlong dst) {
    *(jbyte *) dst = byte;
}

