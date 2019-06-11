package org.mozilla.deepspeech.utils;

import org.jetbrains.annotations.NotNull;
import org.mozilla.deepspeech.exception.buffer.BufferReadonlyException;
import org.mozilla.deepspeech.exception.buffer.IncorrectBufferByteOrderException;
import org.mozilla.deepspeech.exception.buffer.IncorrectBufferTypeException;
import org.mozilla.deepspeech.exception.buffer.UnexpectedBufferCapacityException;

import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Contains buffer related Utility functions
 */
public class BufferUtils {

    /**
     * Checks the byte buffer if it is usable by a native JNI function.
     *
     * @param buffer the buffer to check
     * @throws UnexpectedBufferCapacityException if the buffer has a capacity smaller to #capacity.
     * @throws IncorrectBufferByteOrderException if the buffer has a byte order different to #byteOrder.
     * @throws IncorrectBufferTypeException      if the buffer is not directly allocated.
     * @throws BufferReadonlyException           if the buffer is read only
     */
    public static void checkByteBuffer(@NotNull ByteBuffer buffer, @NotNull ByteOrder expectedByteOrder, long expectedCapacity)
            throws UnexpectedBufferCapacityException, IncorrectBufferByteOrderException, IncorrectBufferTypeException, BufferReadonlyException {
        if (!buffer.isDirect()) {
            throw new IncorrectBufferTypeException("Incorrect buffer type. Expected a directly allocated buffer but received a non direct buffer!");
        }

        if (buffer.isReadOnly()) {
            throw new BufferReadonlyException();
        }

        if (buffer.capacity() < expectedCapacity) {
            throw new UnexpectedBufferCapacityException("Buffer is expected to have at least" + expectedCapacity + " bytes.");
        }

        ByteOrder bufferByteOrder = buffer.order();

        if (bufferByteOrder != expectedByteOrder) {
            throw new IncorrectBufferByteOrderException("Expected a buffer order of " + expectedByteOrder + " but received " + bufferByteOrder);
        }
    }

    /**
     * Method of the address method in the DirectByteBuffer class
     */
    @NotNull
    private static final Method addressMethod;

    static {
        try {
            ClassLoader clazz = BufferUtils.class.getClassLoader();
            if (clazz == null)
                throw new NullPointerException();
            addressMethod = clazz.loadClass("java.nio.DirectByteBuffer").getDeclaredMethod("address");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static long getBufferAddress(@NotNull ByteBuffer buffer) {
        if (!buffer.isDirect()) {
            throw new IncorrectBufferTypeException("Incorrect buffer type. Expected a directly allocated buffer but received a non direct buffer!");
        }
        try {
            addressMethod.setAccessible(true);
            return (Long) addressMethod.invoke(buffer);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
