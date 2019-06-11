package org.mozilla.deepspeech.exception;

/**
 * An exception that is thrown in native JNI methods (or their wrapper).
 */
public abstract class NativeException extends RuntimeException {

    /**
     * @param message exception message
     */
    public NativeException(String message) {
        super(message);
    }

}
