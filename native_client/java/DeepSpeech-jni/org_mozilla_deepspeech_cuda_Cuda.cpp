#include "org_mozilla_deepspeech_cuda_Cuda.h"

jint Java_org_mozilla_deepspeech_cuda_Cuda_nCudaGetDevice(JNIEnv *env, jclass, jobject devicePointer) {
    int *ptr = (int *) env->GetDirectBufferAddress(devicePointer);
    return cudaGetDevice(ptr);
}

jint Java_org_mozilla_deepspeech_cuda_Cuda_nCudaGetDeviceProperties(JNIEnv *env, jclass, jobject dataOut,
                                                                    jint device) {
    size_t size =  256 * sizeof(char) +
                   16 * sizeof(char) +
                   sizeof(long) +
                   sizeof(long) +
                   sizeof(int) +
                   sizeof(int) +
                   sizeof(long) +
                   sizeof(int) +
                   sizeof(int) * 3 +
                   sizeof(int) * 3 +
                   sizeof(long) +
                   sizeof(int) +
                   sizeof(int) +
                   sizeof(int) +
                   sizeof(long) +
                   sizeof(int) +
                   sizeof(int) +
                   sizeof(int) +
                   sizeof(int) +
                   sizeof(int) +
                   sizeof(int);
    printf("size: %zu\n", size);
    fflush(stdout);
    auto *exchangeBuffer = (jbyte *) malloc(
           size
    ); // Size needed for exchange buffer
    auto *bufPtr = exchangeBuffer;
    struct cudaDeviceProp prop{};
    cudaError_t error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaError_t::cudaSuccess) {
        return error;
    }
    memcpy(bufPtr, prop.name, 256 * sizeof(char)); // Copy name char[256]
    bufPtr += 256 * sizeof(char);
    memcpy(bufPtr, &prop.uuid, 16 * sizeof(char));
    bufPtr += 16 * sizeof(char);
    memcpy(bufPtr, &prop.totalGlobalMem, sizeof(size_t)); // copy glob mem long
    bufPtr += sizeof(long);
    memcpy(bufPtr, &prop.sharedMemPerBlock, sizeof(size_t)); // copy shared mem long
    bufPtr += sizeof(long);
    memcpy(bufPtr, &prop.regsPerBlock, sizeof(int)); // copy regs per block int
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.warpSize, sizeof(int)); // copy warp size int
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.memPitch, sizeof(long)); // copy mem pitch long
    bufPtr += sizeof(long);
    memcpy(bufPtr, &prop.maxThreadsPerBlock, sizeof(int)); // copy max threads per block int
    bufPtr += sizeof(int);
    memcpy(bufPtr, prop.maxThreadsDim, 3 * sizeof(int)); // copy maxThreadsDim int[3]
    bufPtr += 3 * sizeof(int);
    memcpy(bufPtr, prop.maxGridSize, 3 * sizeof(int)); // copy maxGridSize int[3]
    bufPtr += 3 * sizeof(int);
    memcpy(bufPtr, &prop.totalConstMem, sizeof(size_t)); // copy totalConstMem long
    bufPtr += sizeof(size_t);
    memcpy(bufPtr, &prop.major, sizeof(int)); // copy major int
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.minor, sizeof(int)); // copy minor int
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.clockRate, sizeof(int)); // copy clockRate ubt
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.textureAlignment, sizeof(long)); // copy textureAlignment long
    bufPtr += sizeof(long);
    memcpy(bufPtr, &prop.deviceOverlap, sizeof(int)); // copy deviceOverlap boolean
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.multiProcessorCount, sizeof(int)); // copy multiProcessorCount int
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.kernelExecTimeoutEnabled, sizeof(int)); // copy kernelExecTimeoutEnabled boolean
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.integrated, sizeof(int)); // copy integrated boolean
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.canMapHostMemory, sizeof(int)); // copy canMapHostMemory boolean
    bufPtr += sizeof(int);
    memcpy(bufPtr, &prop.computeMode, sizeof(int)); // copy canMapHostMemory int

    void **bufferDst = (void **) env->GetDirectBufferAddress(dataOut);
    *bufferDst = exchangeBuffer;

    return cudaError_t::cudaSuccess;
}
