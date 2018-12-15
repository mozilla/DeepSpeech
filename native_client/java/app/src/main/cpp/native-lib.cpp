#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_deepspeech_mozilla_org_deepspeech_DeepSpeechActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
