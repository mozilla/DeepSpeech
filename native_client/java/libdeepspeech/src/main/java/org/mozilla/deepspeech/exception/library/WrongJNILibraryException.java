package org.mozilla.deepspeech.exception.library;

import org.jetbrains.annotations.NotNull;
import org.mozilla.deepspeech.DeepSpeech;

/**
 * Used when the wrong jni library was loaded. That means that a different jni configuration is expected.
 */
public class WrongJNILibraryException extends LibraryLoadingException {

    /**
     * @param expected the expected configuration
     * @param actual the actual configuration
     */
    public WrongJNILibraryException(@NotNull DeepSpeech.JNIConfiguration expected, @NotNull DeepSpeech.JNIConfiguration actual) {
        super("Loaded wrong JNI library. Expected configuration: " + expected + " but got " + actual + "!");
    }
}
