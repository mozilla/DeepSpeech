#include "org_mozilla_deepspeech_DeepSpeech.h"

jint
Java_org_mozilla_deepspeech_DeepSpeech_nCreateModel(JNIEnv *env, jclass, jstring modelPath,
                                                    jlong nCep,
                                                    jlong nContext, jstring alphabetConfigPath,
                                                    jlong beamWidth,
                                                    jobject modelStatePtr) {
    jboolean isModelPathCopy, isAlphaBetCopy;
    ModelState *ptr = nullptr;
    auto modelPathCStr = (char *) env->GetStringUTFChars(modelPath, &isModelPathCopy);
    auto alphaBetCStr = (char *) env->GetStringUTFChars(alphabetConfigPath, &isAlphaBetCopy);

    jint state = DS_CreateModel(modelPathCStr, static_cast<unsigned int>(nCep),
                                static_cast<unsigned int>(nContext), alphaBetCStr,
                                static_cast<unsigned int>(beamWidth),
                                &ptr);
    auto *bufferPtr = (jlong *) (env->GetDirectBufferAddress(modelStatePtr));

    bufferPtr[0] = reinterpret_cast<jlong>(ptr);

    if (isModelPathCopy == JNI_TRUE) {
        env->ReleaseStringUTFChars(modelPath, modelPathCStr);
    }
    return state;
}

void Java_org_mozilla_deepspeech_DeepSpeech_destroyModel(JNIEnv *, jclass, jlong modelPtr) {
    DS_DestroyModel(reinterpret_cast<ModelState *>(modelPtr));
}

jint
Java_org_mozilla_deepspeech_DeepSpeech_enableDecoderWithLM(JNIEnv *env, jclass, jlong modelStatePtr,
                                                           jstring alphaBetConfigPath,
                                                           jstring lmPath,
                                                           jstring triePath, jfloat alpha,
                                                           jfloat beta) {
    jboolean isAlphabetStrCopy, isLmPathCopy, isTriePathCopy;
    auto alphaBetConfigPathCStr = const_cast<char *>(env->GetStringUTFChars(alphaBetConfigPath,
                                                                            &isAlphabetStrCopy));
    auto lmPathCStr = const_cast<char *>(env->GetStringUTFChars(lmPath, &isLmPathCopy));
    auto triePathCStr = const_cast<char *>(env->GetStringUTFChars(triePath, &isTriePathCopy));

    jint status = DS_EnableDecoderWithLM((ModelState *) modelStatePtr, alphaBetConfigPathCStr,
                                         lmPathCStr, triePathCStr,
                                         alpha, beta);

    if (isAlphabetStrCopy == JNI_TRUE) {
        env->ReleaseStringUTFChars(alphaBetConfigPath, alphaBetConfigPathCStr);
    }
    if (isLmPathCopy == JNI_TRUE) {
        env->ReleaseStringUTFChars(lmPath, lmPathCStr);
    }
    if (isTriePathCopy == JNI_TRUE) {
        env->ReleaseStringUTFChars(triePath, triePathCStr);
    }

    return status;
}

jstring
Java_org_mozilla_deepspeech_DeepSpeech_nSpeechToText(JNIEnv *env, jclass, jlong modelStatePtr,
                                                     jobject audioBuffer, jlong numSamples,
                                                     jlong sampleRate) {
    auto *array = (short *) (env->GetDirectBufferAddress(audioBuffer));
    char *cStr = DS_SpeechToText((ModelState *) modelStatePtr, array,
                                 static_cast<unsigned int>(numSamples),
                                 (unsigned int) sampleRate);
    if (cStr == nullptr) {
        return nullptr;
    }
    jstring str = env->NewStringUTF(cStr);
    DS_FreeString(cStr);
    return str;
}

jstring
Java_org_mozilla_deepspeech_DeepSpeech_speechToTextUnsafe(JNIEnv *env, jclass, jlong modelStatePtr,
                                                     jlong audioBuffer, jlong numSamples,
                                                     jlong sampleRate) {
    auto *array = (short *) (audioBuffer);
    char *cStr = DS_SpeechToText((ModelState *) modelStatePtr, array,
                                 static_cast<unsigned int>(numSamples),
                                 (unsigned int) sampleRate);
    if (cStr == nullptr) {
        return nullptr;
    }
    jstring str = env->NewStringUTF(cStr);
    DS_FreeString(cStr);
    return str;
}

jlong
Java_org_mozilla_deepspeech_DeepSpeech_nSpeechToTextWithMetadata(JNIEnv *env, jclass,
                                                                 jlong modelStatePtr,
                                                                 jobject audioBuffer,
                                                                 jlong bufferSize,
                                                                 jlong sampleRate) {
    auto *array = static_cast<short *>(env->GetDirectBufferAddress(audioBuffer));
    auto metaPtr = reinterpret_cast<jlong>(DS_SpeechToTextWithMetadata((ModelState *) modelStatePtr,
                                                                       array,
                                                                       static_cast<unsigned int>(bufferSize),
                                                                       static_cast<unsigned int>(sampleRate)));
    return metaPtr;
}
jlong
Java_org_mozilla_deepspeech_DeepSpeech_speechToTextWithMetadataUnsafe(JNIEnv *, jclass,
                                                                 jlong modelStatePtr,
                                                                 jlong audioBuffer,
                                                                 jlong bufferSize,
                                                                 jlong sampleRate) {
    auto *array = (short *)audioBuffer;
    auto metaPtr = reinterpret_cast<jlong>(DS_SpeechToTextWithMetadata((ModelState *) modelStatePtr,
                                                                       array,
                                                                       static_cast<unsigned int>(bufferSize),
                                                                       static_cast<unsigned int>(sampleRate)));
    return metaPtr;
}

jint Java_org_mozilla_deepspeech_DeepSpeech_nSetupStream(JNIEnv *env, jclass, jlong modelStatePtr,
                                                         jlong preAllocFrames, jlong sampleRate,
                                                         jobject streamPtr) {
    StreamingState *pStreamingState;

    jint status = DS_SetupStream((ModelState *) modelStatePtr,
                                 static_cast<unsigned int>(preAllocFrames),
                                 static_cast<unsigned int>(sampleRate), &pStreamingState);
    auto p = (StreamingState **) env->GetDirectBufferAddress(streamPtr);
    *p = pStreamingState;
    return status;
}

void Java_org_mozilla_deepspeech_DeepSpeech_nFeedAudioContent(JNIEnv *env, jclass, jlong streamPtr,
                                                              jobject audioBuffer,
                                                              jlong bufferSize) {
    auto *test = static_cast<short *>(env->GetDirectBufferAddress(audioBuffer));
    DS_FeedAudioContent((StreamingState *) streamPtr, test, static_cast<unsigned int>(bufferSize));
}

jstring
Java_org_mozilla_deepspeech_DeepSpeech_intermediateDecode(JNIEnv *env, jclass, jlong streamPtr) {
    char *cString = DS_IntermediateDecode((StreamingState *) streamPtr);
    jstring str = env->NewStringUTF(cString);
    DS_FreeString(cString);
    return str;
}

jstring Java_org_mozilla_deepspeech_DeepSpeech_finishStream(JNIEnv *env, jclass, jlong streamPtr) {
    char *cString = DS_FinishStream((StreamingState *) streamPtr);
    size_t cStrLen = strlen(cString);
    jstring str = env->NewString(reinterpret_cast<const jchar *>(cString),
                                 static_cast<jsize>(cStrLen));
    DS_FreeString(cString);
    return str;
}

jlong
Java_org_mozilla_deepspeech_DeepSpeech_finishStreamWithMetadata(JNIEnv *, jclass, jlong streamPtr) {
    return reinterpret_cast<jlong>(DS_FinishStreamWithMetadata((StreamingState *) streamPtr));
}

void Java_org_mozilla_deepspeech_DeepSpeech_discardStream(JNIEnv *, jclass, jlong streamPtr) {
    DS_DiscardStream((StreamingState *) streamPtr);
}

void Java_org_mozilla_deepspeech_DeepSpeech_freeMetadata(JNIEnv *, jclass, jlong metaPtr) {
    DS_FreeMetadata((Metadata *) metaPtr);
}

void Java_org_mozilla_deepspeech_DeepSpeech_printVersions(JNIEnv *, jclass) {
    DS_PrintVersions();
}

jint Java_org_mozilla_deepspeech_DeepSpeech_nGetConfiguration(JNIEnv *, jclass) {
#ifdef CPU_BUILD_CONFIG
    return BuildConfiguration::CPU;
#else
#ifdef CUDA_BUILD_CONFIG
    return BuildConfiguration::CUDA;
#else
    return BuildConfiguration::INVALID; // This should never be returned
#endif
#endif
}