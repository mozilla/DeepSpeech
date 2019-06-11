#ifndef DEEPSPEECH_JNI_ORG_MOZILLA_DEEPSPEECH_CUDA_CUDA_H
#define DEEPSPEECH_JNI_ORG_MOZILLA_DEEPSPEECH_CUDA_CUDA_H

#include <jni.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_cuda_Cuda_nCudaGetDevice
        (JNIEnv *, jclass, jobject);

JNIEXPORT jint JNICALL Java_org_mozilla_deepspeech_cuda_Cuda_nCudaGetDeviceProperties
        (JNIEnv *, jclass, jobject, jint);


#ifdef __cplusplus
}
#endif

#endif //DEEPSPEECH_JNI_ORG_MOZILLA_DEEPSPEECH_CUDA_CUDA_H
