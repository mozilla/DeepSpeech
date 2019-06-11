package org.mozilla.deepspeech.exception.library;

import org.mozilla.deepspeech.DeepSpeech;

/**
 * Exception thrown when an invalid jni library is detected. This means that it has not been built properly.
 */
public class InvalidJNILibraryException extends LibraryLoadingException {

    /**
     * Used when the configuration could not be resolved to a parent java enum constant of {@link org.mozilla.deepspeech.DeepSpeech.JNIConfiguration}
     *
     * @param configuration the native enum value
     */
    public InvalidJNILibraryException(int configuration) {
        super("The used jni library has an unknown configuration: " + configuration + ". This means that the library has not been built properly!");
    }

    /**
     * Used when the configuration is resolved but invalid. (eg. {@link org.mozilla.deepspeech.DeepSpeech.JNIConfiguration#NAN})
     *
     * @param configuration the enum constant of {@link org.mozilla.deepspeech.DeepSpeech.JNIConfiguration}
     */
    public InvalidJNILibraryException(DeepSpeech.JNIConfiguration configuration) {
        super("The used jni library has an invalid configuration: " + configuration + ". This means that the library has not been built properly!");
    }
}
