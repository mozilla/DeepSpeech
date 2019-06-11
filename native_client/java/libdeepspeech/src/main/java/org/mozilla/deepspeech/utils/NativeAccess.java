package org.mozilla.deepspeech.utils;

import org.jetbrains.annotations.NotNull;
import org.mozilla.deepspeech.doc.NativeType;

/**
 * Functions to access native memory. * Similar to {@link sun.misc.Unsafe}.
 * Unsafe is not used to avoid class loader issues.
 */
public class NativeAccess {

    /**
     * Native size used for the integer. May be 2 bytes, 4 bytes or 8 bytes.
     */
    public static final int NATIVE_INT_SIZE;

    /**
     * Native size used for a memory address (pointer). May be 4 bytes or 8 bytes.
     */
    public static final int NATIVE_POINTER_SIZE;

    static {
        NATIVE_INT_SIZE = getNativeIntSize();
        NATIVE_POINTER_SIZE = getNativePointerSize();
    }

    /**
     * Converts a c-string into a java string. The null char is not included in the java string.
     *
     * @param cString the pointer that points to the first character of the native character sequence that must be terminated by a null char.
     * @return the converted java string
     */
    @NotNull
    public static native String getNativeCString(@NativeType("char *") long cString);

    /**
     * @param charPointer a pointer pointing to a c character (1 byte). (NO 2 BYTE JAVA CHARACTER)
     * @return the value of the de-referenced pointer
     */
    public static native char getNativeChar(@NativeType("char *") long charPointer);

    /**
     * @param shortPointer a pointer pointing to a short
     * @return the value of the de-referenced pointer
     */
    public static native short getNativeShort(@NativeType("short *") long shortPointer);

    /**
     * @param intPointer a pointer pointing to an int
     * @return the value of the de-referenced pointer
     */
    public static native int getNativeInt(@NativeType("int *") long intPointer);

    /**
     * @param longPointer a pointer pointing to a long
     * @return the value of the de-referenced pointer
     */
    public static native long getNativeLong(@NativeType("long *") long longPointer);

    /**
     * @param pointerPointer the pointer pointing to the pointer to be de-referenced
     * @return the value of the de-referenced pointer as void pointer
     */
    @NativeType("void *")
    public static native long getNativePointer(@NativeType("void **") long pointerPointer);

    /**
     * @param doublePointer a pointer pointing to a double
     * @return the value of the de-referenced pointer
     */
    public static native double getNativeDouble(@NativeType("double *") long doublePointer);

    /**
     * @param floatPointer a pointer pointing to a float
     * @return the value of the de-referenced pointer
     */
    public static native float getNativeFloat(@NativeType("float *") long floatPointer);

    /**
     * @param booleanPointer a pointer pointing to a c++ bool. As a java boolean is only 1 byte big, the interpretation of the logical value is "{@code static_cast<jboolean>((*(int *) booleanPointer) != 0);}"
     * @return the value of the de-referenced pointer
     */
    public static native boolean getNativeBoolean(@NativeType("bool *") long booleanPointer);

    /**
     * Unexposed function. Use constant {@link #NATIVE_INT_SIZE}
     *
     * @return sizeof(int)
     */
    private static native int getNativeIntSize();

    /**
     * Unexposed function. Use constant {@link #NATIVE_POINTER_SIZE}
     *
     * @return sizeof(void *)
     */
    private static native int getNativePointerSize();

    /**
     * Allocates #bytes on the heap and returns the pointer. The byte contents of the allocated memory are all set to zero. <a href="http://www.cplusplus.com/reference/cstdlib/free/">See c++ documentation of native function</a>
     * @param bytes the amount of bytes to allocate
     * @return the pointer to the natively allocated heap memory. Returns a null pointer of #bytes amount of bytes could not be allocated continuously.
     */
    public static native long allocateMemory(long bytes);

    /**
     * Frees the dynamically allocated memory that this pointer points to. See posix's free function <a href="http://www.cplusplus.com/reference/cstdlib/free/">See c++ documentation of native function</a>
     * @param pointer the pointer pointing to the dynamic memory to be freed.
     */
    public static native void freeMemory(@NativeType("void *") long pointer);

    /**
     * Copies #numBytes amount of bytes of the memory region behind the #source pointer onto the memory region behind the #destionation pointer. <a href="http://www.cplusplus.com/reference/cstring/memcpy/">See c++ documentation of native function</a>
     * @param destination the destination
     * @param source the source
     * @param numBytes the amount of bytes to copy.
     */
    public static native void copyMemory(long destination, long source, long numBytes);

    /**
     * Writes the specified byte onto the memory behind the address
     * @param b the byte
     * @param address the destination memory pointer
     */
    public static native void writeByte(byte b, long address);
}