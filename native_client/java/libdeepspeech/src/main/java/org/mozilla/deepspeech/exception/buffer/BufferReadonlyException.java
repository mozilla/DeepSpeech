package org.mozilla.deepspeech.exception.buffer;

/**
 * An exception thrown when a buffer is expected to be writable but a read-only buffer is received.
 */
public class BufferReadonlyException extends InvalidBufferException {

    public BufferReadonlyException() {
        super("Buffer is expected to be writable but received a read-only buffer instance.");
    }
}
