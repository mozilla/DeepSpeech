package org.mozilla.deepspeech.exception.cuda;

import org.mozilla.deepspeech.exception.NativeException;

/**
 * An exception thrown by a cuda function
 */
public class CudaException extends NativeException {

    /**
     * @param message exception message
     */
    public CudaException(String message) {
        super(message);
    }
}
