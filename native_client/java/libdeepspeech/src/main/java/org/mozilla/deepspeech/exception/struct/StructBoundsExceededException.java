package org.mozilla.deepspeech.exception.struct;

import org.mozilla.deepspeech.exception.NativeException;

/**
 * An exception thrown when a native structure was accessed illegally meaning out of the allocated memory region dedicated to the structure.
 */
public class StructBoundsExceededException extends NativeException {

    public StructBoundsExceededException(long triedByteOffset, long structSize) {
        super(triedByteOffset < 0 ? "Bounds of native struct were undershot. Tried to access pre struct start memory. Tried offset: " + triedByteOffset + "." : "Bounds of native struct exceeded. Tried byte offset: " + triedByteOffset + ". Struct size is " + structSize);
    }
}
