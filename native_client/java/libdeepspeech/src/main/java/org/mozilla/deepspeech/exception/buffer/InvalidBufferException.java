package org.mozilla.deepspeech.exception.buffer;

import org.mozilla.deepspeech.exception.NativeException;

/**
 * An exception thrown when an invalid buffer is received.
 */
public class InvalidBufferException extends NativeException {
    /**
     * @param message exception message
     */
    public InvalidBufferException(String message) {
        super(message);
    }
}
