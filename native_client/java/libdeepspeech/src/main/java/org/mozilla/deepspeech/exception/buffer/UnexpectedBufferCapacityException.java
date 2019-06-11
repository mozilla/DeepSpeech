package org.mozilla.deepspeech.exception.buffer;

/**
 * An exception thrown when a buffer has an incorrect buffer size compared to what is specified or needed eg. when not enough capacity was allocated.
 */
public class UnexpectedBufferCapacityException extends InvalidBufferException {

    /**
     * @param message exception message
     */
    public UnexpectedBufferCapacityException(String message) {
        super(message);
    }
}
