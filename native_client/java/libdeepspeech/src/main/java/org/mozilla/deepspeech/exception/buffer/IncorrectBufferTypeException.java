package org.mozilla.deepspeech.exception.buffer;

/**
 * An exception thrown when a buffer has an unexpected Type eg. when a direct buffer is expected by a native method buf instead receives a HeapBuffer that it cannot work with.
 */
public class IncorrectBufferTypeException extends InvalidBufferException {

    /**
     * @param message exception message
     */
    public IncorrectBufferTypeException(String message) {
        super(message);
    }
}
