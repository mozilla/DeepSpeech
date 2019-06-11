package org.mozilla.deepspeech.exception.library;

import org.mozilla.deepspeech.exception.NativeException;

/**
 * An exception thrown because of native library loading
 */
public class LibraryLoadingException extends NativeException {

    /**
     * @param message exception message
     */
    public LibraryLoadingException(String message) {
        super(message);
    }
}
