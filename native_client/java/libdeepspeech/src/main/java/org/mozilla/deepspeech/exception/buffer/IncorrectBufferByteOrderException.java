package org.mozilla.deepspeech.exception.buffer;

/**
 * An exception thrown by a native method that retrieves a buffer when the buffer has an unexpected byte order.
 * @see java.nio.ByteOrder
 */
public class IncorrectBufferByteOrderException extends InvalidBufferException {

    /**
     * @param message exception message
     */
    public IncorrectBufferByteOrderException(String message) {
        super(message);
    }
}
